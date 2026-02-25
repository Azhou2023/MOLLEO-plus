import os
import requests
import yaml
import random
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import tdc
from tdc.generation import MolGen
from main.utils.chem import *
import math

from .boltz import calculate_boltz
from .docking import calculate_docking
from torch import multiprocessing as mp
from queue import Empty

from kubernetes import config, client

num_gpus = torch.cuda.device_count()
API_URL = "https://andrew-boltz-api.nrp-nautilus.io/predict_affinity"
use_nautilus = True

class Objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False)) # increasing order
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls

def get_ready_pod_count(namespace, deployment_name):

    try:
        config.load_kube_config()
    except config.ConfigException:
        config.load_incluster_config()

    api_instance = client.AppsV1Api()

    try:
        deployment = api_instance.read_namespaced_deployment(
            name=deployment_name,
            namespace=namespace
        )

        ready_count = deployment.status.ready_replicas or 0
        print(f"Deployment: {deployment_name}")
        print(f"Total Desired: {deployment.status.replicas}")
        print(f"Ready & Running: {ready_count}")
        
        return ready_count

    except client.exceptions.ApiException as e:
        if e.status == 404:
            print(f"Error: Deployment '{deployment_name}' not found in namespace '{namespace}'.")
        else:
            print(f"API Error: {e}")
        return 0

def calculate_boltz_nautilus(protein_name, ligand_smiles, idx):
    print(f"\n[Worker {str(idx)}] Sending job for {ligand_smiles}...", flush=True)
    
    query_params = {
        "protein_name": protein_name,
        "ligand": ligand_smiles
    }
    worker_lifetime = time.time()
    while True:
        try:
            before = time.time()
            response = requests.post(API_URL, params=query_params, headers={'Connection': 'close'})
            after = time.time()
            print(response)
            if response.status_code == 200:
                result = response.json()
                print(f"\n[Worker {str(idx)}] Success")
                print(result)
                print(f"[Worker {str(idx)}]Took: {str(after-before)} seconds", flush=True)
                time.sleep(3)
                return result["affinity"]
            elif response.status_code == 429 or response.status_code == 503:
                sleep_time = random.uniform(0.5, 2.0)
                print(f"[Worker {str(idx)}] Busy: Retrying", flush=True)
                current_lifetime = time.time()
                if worker_lifetime - current_lifetime > 3600:
                    print(f"[Worker {str(idx)}] Worker has been stalled for {str(worker_lifetime-current_lifetime)} seconds. Exiting.", flush=True)
                    return 0
                time.sleep(sleep_time)
            else:
                print(f"[Worker {str(idx)}] Error {response.status_code}: {response.text}", flush=True)
                return 0
        except Exception as e:
            print(f"\n[Worker {str(idx)}] Connection failed: {str(e)}", flush=True)        
            time.sleep(3)
            return 0


def gpu_worker(gpu_id, task_q, result_q, evaluator, boltz_cache):
    while True:
        try:
            idx, x, val = task_q.get(timeout=3)
            if x is None:
                result_q.put((idx, x, -100))
                print("Buffer is full")
            elif val is not None:
                result_q.put((idx, x, val))
                print("Already in buffer")
            else:
                y = 0
                max_steps = 5
                i = 0
                while y==0 and i < max_steps:
                    if boltz_cache and x in boltz_cache:
                        y = -float(boltz_cache[x])
                        break
                    if gpu_id >= num_gpus:
                        y = -float(calculate_boltz_nautilus(evaluator, x, gpu_id))
                    else:
                        y = -float(calculate_boltz(evaluator, x, gpu_id))
                    i+=1
                result_q.put((idx, x, y))
                print(f"\nWorker {gpu_id} produced result: {str((x, y))}", flush=True)
        except Empty:
            break
        

class Oracle:
    def __init__(self, args=None, seed=None, mol_buffer={}):
        self.name = None
        self.evaluator = None
        self.task_label = None
        self.seed = seed
        if args is None:
            self.max_oracle_calls = 10000
            self.freq_log = 100
        else:
            self.args = args
            self.max_oracle_calls = args.max_oracle_calls
            self.freq_log = args.freq_log

        self.mol_buffer = mol_buffer
        self.sa_scorer = tdc.Oracle(name = 'SA')
        self.diversity_evaluator = tdc.Evaluator(name = 'Diversity')
        self.last_log = 0

        self.boltz_cache = None
        self.oracle_name=None

    def parallel_oracle(self, inputs):
        print(inputs)
        print("Number of GPUs available: " + str(num_gpus))
        assert num_gpus > 0, "No GPUs available"

        ctx = mp.get_context("spawn")
        task_q = ctx.Queue()
        result_q = ctx.Queue()

        # enqueue tasks
        num_new = 0
        for i, x in enumerate(inputs):
            if len(self.mol_buffer) + num_new >= self.max_oracle_calls:
                task_q.put((i, None, None))
                continue
            if x in self.mol_buffer:
                task_q.put((i, x, self.mol_buffer[x][0]))
            else:
                task_q.put((i, x, None))
                num_new += 1

        # start one worker per GPU
        procs = []
        num_nautilus = get_ready_pod_count("spatiotemporal-decision-making", "andrew-boltz-api")
        num_workers = num_gpus + num_nautilus if use_nautilus else num_gpus
        print(f"{str(num_workers)} workers available ({str(num_gpus)} local, {str(num_nautilus)} on Nautilus)")
        for gpu_id in range(num_workers):
            p = ctx.Process(target=gpu_worker, args=(gpu_id, task_q, result_q, self.evaluator, self.boltz_cache))
            p.start()
            procs.append(p)
            time.sleep(0.1)

        # collect results
        results = [None] * len(inputs)
        buffer_length = len(self.mol_buffer)
        num_added = 0
        for i in range(len(inputs)):
            idx, x, y = result_q.get()
            results[idx] = y
            if x not in self.mol_buffer and y!=-100: 
                self.mol_buffer[x] = [y, buffer_length+num_added+1]
                num_added += 1
        for p in procs:
            p.join()
        print("RESULTS: " + str(results))
        print(self.mol_buffer)
        return results


    @property
    def budget(self):
        return self.max_oracle_calls

    def assign_evaluator(self, evaluator):
        self.evaluator = evaluator

    def sort_buffer(self):
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def save_result(self, suffix=None):

        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'results/' + suffix + '.yaml')

        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)


    def log_intermediate(self, mols=None, scores=None, finish=False):

        if finish:
            temp_top100 = list(self.mol_buffer.items())[:100]
            smis = [item[0] for item in temp_top100]
            scores = [item[1][0] for item in temp_top100]
            n_calls = self.max_oracle_calls
        else:
            if mols is None and scores is None:
                if len(self.mol_buffer) <= self.max_oracle_calls:
                    # If not spefcified, log current top-100 mols in buffer
                    temp_top100 = list(self.mol_buffer.items())[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = len(self.mol_buffer)
                else:
                    results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))[:self.max_oracle_calls]
                    temp_top100 = sorted(results, key=lambda kv: kv[1][0], reverse=True)[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = self.max_oracle_calls
            else:
                # Otherwise, log the input moleucles
                smis = [Chem.MolToSmiles(m, canonical=True) for m in mols]
                n_calls = len(self.mol_buffer)

        # Uncomment this line if want to log top-10 moelucles figures, so as the best_mol key values.
        # temp_top10 = list(self.mol_buffer.items())[:10]

        avg_top1 = np.max(scores)
        avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
        avg_top100 = np.mean(scores)
        avg_sa = np.mean(self.sa_scorer(smis))
        diversity_top100 = self.diversity_evaluator(smis)


        print(f'{n_calls}/{self.max_oracle_calls} | '
                f'avg_top1: {avg_top1:.3f} | '
                f'avg_top10: {avg_top10:.3f} | '
                f'avg_top100: {avg_top100:.3f} | '
                f'avg_sa: {avg_sa:.3f} | '
                f'div: {diversity_top100:.3f}')

        print({
            "avg_top1": avg_top1,
            "avg_top10": avg_top10,
            "avg_top100": avg_top100,
            "auc_top1": top_auc(self.mol_buffer, 1, finish, self.freq_log, self.max_oracle_calls),
            "auc_top10": top_auc(self.mol_buffer, 10, finish, self.freq_log, self.max_oracle_calls),
            "auc_top100": top_auc(self.mol_buffer, 100, finish, self.freq_log, self.max_oracle_calls),
            "avg_sa": avg_sa,
            "diversity_top100": diversity_top100,
            "n_oracle": n_calls,
        })




    def __len__(self):
        return len(self.mol_buffer)

    def get_docking_data(self, smiles, protein):
            try:
                response = requests.get(f"https://west.ucsd.edu/llm_project/?endpoint=run_docking&smiles={smiles}&target={protein}")
                if response.status_code == 200:
                    data = response.json()
                    if 'error' in data:
                        return 0
                    return data["binding_affinity"]
                else:
                    time.sleep(60)
                    return 0
            except Exception as e:
                print(e)
                time.sleep(60)
                return 0

    def score_smi(self, smi, device):
        """
        Function to score one molecule

        Argguments:
            smi: One SMILES string represnets a moelcule.

        Return:
            score: a float represents the property of the molecule.
        """
        if len(self.mol_buffer) > self.max_oracle_calls:
            return 0
        if smi in self.mol_buffer:
            print("Already in buffer", flush=True)
            pass
        else:
            print(smi, flush=True)
            fitness = -float(calculate_boltz(self.evaluator, smi, device))
            # fitness = -float(calculate_docking(self.evaluator, smi))
            if fitness < 0.0: fitness = 0.0
            print(fitness, flush=True)
            #print(fitness, type(fitness))
            if math.isnan(fitness):
                fitness = 0
            self.mol_buffer[smi] = [fitness, len(self.mol_buffer)+1]
        return self.mol_buffer[smi][0]

    def __call__(self, smiles_lst):
        """
        Score
        """
        if type(smiles_lst) == list:
            print(len(smiles_lst), flush=True)
            
            score_list = self.parallel_oracle(smiles_lst)
            self.sort_buffer()
            self.log_intermediate()
            self.last_log = len(self.mol_buffer)
            self.save_result(self.task_label)
        else:  ### a string of SMILES
            score_list = self.score_smi(smiles_lst)
            if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                self.sort_buffer()
                self.log_intermediate()
                self.last_log = len(self.mol_buffer)
                run_name = self.args.run_name + "_" + str(self.seed)
                self.save_result(self.task_label)
        return score_list

    @property
    def finish(self):
        return len(self.mol_buffer) >= self.max_oracle_calls


class BaseOptimizer:

    def __init__(self, args=None, seed=None):
        self.model_name = args.mol_lm
        self.args = args
        self.seed = seed
        print(self.args.run_name, flush=True)
        self.n_jobs = args.n_jobs
        # self.pool = joblib.Parallel(n_jobs=self.n_jobs)
        self.smi_file = args.smi_file
        self.oracle = Oracle(args=self.args, seed=self.seed)
        if self.smi_file is not None:
            self.all_smiles = self.load_smiles_from_file(self.smi_file)
        else:
            data = MolGen(name = 'ZINC')
            print(data)
            self.all_smiles = data.get_data()['smiles'].tolist()
            # self.all_smiles = []
            # with open(f"data/{args.oracles[0]}.txt", "r") as file:
            #     for line in file:
            #         ligand = line[:-1]
            #         self.all_smiles.append(ligand)


        self.sa_scorer = tdc.Oracle(name = 'SA')
        self.diversity_evaluator = tdc.Evaluator(name = 'Diversity')
        self.filter = tdc.chem_utils.oracle.filter.MolFilter(filters = ['PAINS', 'SureChEMBL', 'Glaxo'], property_filters_flag = False)

    # def load_smiles_from_file(self, file_name):
    #     with open(file_name) as f:
    #         return self.pool(delayed(canonicalize)(s.strip()) for s in f)

    def sanitize(self, smiles_list, score_list=None):
        new_smiles_list = []
        smiles_set = set()
        for smiles in smiles_list:
            if smiles is not None:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if not mol:
                        continue
                    smiles = Chem.MolToSmiles(mol, canonical=True)
                    if smiles is not None and smiles not in smiles_set:
                        smiles_set.add(smiles)
                        new_smiles_list.append(smiles)
                except ValueError:
                    print('bad smiles')
        return new_smiles_list

    def sort_buffer(self):
        self.oracle.sort_buffer()

    def log_intermediate(self, mols=None, scores=None, finish=False):
        self.oracle.log_intermediate(mols=mols, scores=scores, finish=finish)

    def log_result(self):

        print(f"Logging final results...")

        # import ipdb; ipdb.set_trace()

        log_num_oracles = [100, 500, 1000, 3000, 5000, 10000]
        assert len(self.mol_buffer) > 0

        results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))
        if len(results) > 10000:
            results = results[:10000]

        results_all_level = []
        for n_o in log_num_oracles:
            results_all_level.append(sorted(results[:n_o], key=lambda kv: kv[1][0], reverse=True))


        # Log batch metrics at various oracle calls
        data = [[log_num_oracles[i]] + self._analyze_results(r) for i, r in enumerate(results_all_level)]
        columns = ["#Oracle", "avg_top100", "avg_top10", "avg_top1", "Diversity", "avg_SA", "%Pass", "Top-1 Pass"]

    def save_result(self, suffix=None):

        print(f"Saving molecules...")

        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'results/' + suffix + '.yaml')

        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)

    def _analyze_results(self, results):
        results = results[:100]
        scores_dict = {item[0]: item[1][0] for item in results}
        smis = [item[0] for item in results]
        scores = [item[1][0] for item in results]
        smis_pass = self.filter(smis)
        if len(smis_pass) == 0:
            top1_pass = -1
        else:
            top1_pass = np.max([scores_dict[s] for s in smis_pass])
        return [np.mean(scores),
                np.mean(scores[:10]),
                np.max(scores),
                self.diversity_evaluator(smis),
                np.mean(self.sa_scorer(smis)),
                float(len(smis_pass) / 100),
                top1_pass]

    def reset(self):
        del self.oracle
        self.oracle = Oracle(args=self.args)

    @property
    def mol_buffer(self):
        return self.oracle.mol_buffer

    @property
    def finish(self):
        return self.oracle.finish

    def _optimize(self, oracle, config):
        raise NotImplementedError



    def optimize(self, oracle, config, project="test"):

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        run_name = self.args.run_name + "_" + str(self.seed)
        self.oracle.task_label = run_name
        if self.seed <= 2:
            with open(f"/home/ubuntu/MOLLEO/init_caches/{oracle}_{str(self.seed)}.yaml", 'r') as file:
                self.oracle.boltz_cache = yaml.safe_load(file)
                print(self.oracle.boltz_cache)

        self._optimize(oracle, config)
        if self.args.log_results:
            self.log_result()
        self.save_result(run_name)
        self.reset()

