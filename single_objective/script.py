import os
import sys
import numpy as np
import yaml
from similarity_clustering import cluster
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import QED
from rdkit.Chem import DataStructs
from rdkit.Chem import Descriptors
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt
from rdkit.Chem import Draw

from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

def analyze_results(run_name, limit=False, llm_only=True, eval_sim=False):
    print("RUN: " + run_name)
    
    num_seeds = 3
    all_llm_ligands = [[] for _ in range(num_seeds)]
    num_errors = 0
    curr_seed = 0
    seeds_passed = 0
    if llm_only:
        with open(f"single_objective/logs/{run_name}.txt", 'r') as log_file:
            for line in log_file:
                if "seed" in line and line.split()[0]=="seed":
                    curr_seed = int(line.split()[1].strip())
                    seeds_passed += 1
                if "1000/1000" in line and seeds_passed == num_seeds:
                    break
                if "LLM-GENERATED:" in line:
                    ligand = line.split()[1].strip()
                    all_llm_ligands[curr_seed].append(ligand)
                if "NUM LLM ERRORS" in line:
                    num_errors += 1
    avg_top_10 = []
    avg_top_10_qed = 0
    avg_top_10_sa = 0
    avg_top_10_max_sim = 0
    better_than_threshold = 0
    
    avg_max_sim_filtered = 0
    unique_mean = []
    
    num_filtered = 0
    filter_mean = 0
    
    pareto_size = 0
    pareto_mean = 0
    pareto_qed_mean = 0
    pareto_sa_mean = 0
    hypervolume = 0
    
    output_mols = []
    morgan = AllChem.GetMorganGenerator(radius=2, fpSize=512)
    for seed, llm_ligands in enumerate(all_llm_ligands):
        cmet = []
        ligands = {}
        cache = {}
        with open(f"single_objective/results/{run_name}_{seed}.yaml", 'r') as file:
            data = yaml.safe_load(file)
            sorted_data = sorted(data, key=lambda k: data[k][1])
            for idx, ligand in enumerate(sorted_data):
                if limit and idx > 255:
                    continue
                if idx < 120:
                    cmet.append(ligand)
                    cache[ligand] = -float(data[ligand][0])
                else:
                    if (llm_only is False or ligand in llm_ligands) and float(data[ligand][0])>0:
                        affin = -float(data[ligand][0])
                        mol = Chem.MolFromSmiles(ligand)
                        qed = QED.qed(mol)
                        sa = sascorer.calculateScore(mol)
                        # mw = Descriptors.MolWt(mol)
                        
                        max_sim = 0
                        if eval_sim:
                            fingerprint = morgan.GetFingerprint(mol)
                            
                            # scaf = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
                            # fingerprint = AllChem.GetMorganFingerprintAsBitVect(scaf, radius=2, nBits=2048)

                            # sim_ligand = ""
                            for cmet_ligand in cmet:
                                cmet_mol = Chem.MolFromSmiles(cmet_ligand)
                                
                                cmet_fingerprint = morgan.GetFingerprint(cmet_mol)

                                # cmet_scaf = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(cmet_mol)
                                # cmet_fingerprint = AllChem.GetMorganFingerprintAsBitVect(cmet_scaf, radius=2, nBits=2048)
                                
                                similarity = DataStructs.TanimotoSimilarity(fingerprint, cmet_fingerprint)
                                
                                if similarity > max_sim:
                                    max_sim = similarity
                                    # sim_ligand = cmet_ligand
                        ligands[ligand] = [affin, qed, sa, max_sim]
        
        if "c-met" in run_name:
            with open(f"/home/ubuntu/MOLLEO/init_caches/c-met_{seed}.yaml", 'w') as file:
                yaml.dump(cache, file)
        elif "brd4" in run_name:
            with open(f"/home/ubuntu/MOLLEO/init_caches/brd4_{seed}.yaml", 'w') as file:
                yaml.dump(cache, file)

        sorted_ligands = sorted(ligands, key=lambda k: ligands[k][0])
        filtered_ligands = [ligand for ligand in sorted_ligands if ligands[ligand][3] < 0.5]
        
        
        c = cluster(sorted_ligands)
        c = sorted(c, key=lambda k: ligands[k][0])
        best_10_cluster = []
        qed = []
        sa = []
        sim = []
        threshold = -11
        num_better_than_threshold = 0
        for i in c[:10]:
            print(i)
            best_10_cluster.append(ligands[i][0])
            qed.append(ligands[i][1])
            sa.append(ligands[i][2])
            sim.append(ligands[i][3])
            if ligands[i][0] < threshold:
                num_better_than_threshold += 1
            
        c_filtered = cluster(filtered_ligands)
        c_filtered = sorted(c_filtered, key=lambda k: ligands[k][0])
        for i in c_filtered[:10]:
            print(i)
        
        avg_top_10.append(np.mean(best_10_cluster))
        avg_top_10_qed += np.mean(qed)
        avg_top_10_sa += np.mean(sa)
        better_than_threshold += num_better_than_threshold
        avg_top_10_max_sim += np.mean(sim)
        
        avg_max_sim_filtered += np.mean([ligands[i][3] for i in filtered_ligands])
        unique_mean.append(np.mean([ligands[i][0] for i in c_filtered[:10]]))
        # print("AVG TOP TEN: " + str(np.mean(best_10)))
        # print("AVG TOP TEN (CLUSTERED): " + str(np.mean(best_10_cluster)))
        # print("BEST: " + str(min(best_10_cluster)))
        # print("STDEV TOP 10 (CLUSTERED): " + str(np.std(best_10_cluster)))
        # print("BEST 10 LIGANDS (CLUSTERED):")
                
        # print("AVG QED (clustered): " + str(np.mean(qed)))
        # print("AVG SA (clustered): " + str(np.mean(sa)))
        # print("STDEV QED: " + str(np.std(qed)))
        # print("AVG MAX SIM: " + str(np.mean(sim)))
        # print("NUMBER OF LLM ERRORS: " + str(num_errors))
        # print("NUM BETTER THAN THRESHOLD: " + str(num_better_than_threshold))
        # print("UNIQUE GENERATION MEAN: " + str(np.mean(sorted(unique)[:10])))
        # print("UNIQUE GENERATION STD: " + str(np.std(sorted(unique)[:10])))
        
        # pareto analysis
        ligands_list = list(ligands.keys())
        score_list = []
        pareto_affins = []
        pareto_qed = []
        pareto_sa = []
        pareto_sim = []
        filtered_mols = []
        for ligand in ligands_list:
            pass_filter = True

            single_score = []
            
            single_score.append(1-(-ligands[ligand][0]/15))
            pareto_affins.append(ligands[ligand][0])
            
            single_score.append(1 - ligands[ligand][1])
            pareto_qed.append(ligands[ligand][1])
            if ligands[ligand][1] < 0.6:
                pass_filter = False
            
            single_score.append((ligands[ligand][2]-1)/9)
            pareto_sa.append(ligands[ligand][2])
            if ligands[ligand][2] > 3.0:
                pass_filter = False
            
            # if "bindingdb" in run_name or eval_sim: 
            #     mol = Chem.MolFromSmiles(ligand)
            #     fingerprint = morgan.GetFingerprint(mol)
            #     max_sim = 0
            #     for cmet_ligand in cmet:
            #         cmet_mol = Chem.MolFromSmiles(cmet_ligand)
            #         cmet_fingerprint = morgan.GetFingerprint(cmet_mol)
            #         similarity = DataStructs.TanimotoSimilarity(fingerprint, cmet_fingerprint)
            #         max_sim = max(max_sim, similarity)
            #     if max_sim > 0.3:
            #         pass_filter = False
                        
            if not eval_sim or max_sim < 0.3: 
                score_list.append(single_score)
            else:
                score_list.append([1.0, 1.0, 1.0])
                
            if pass_filter:
                filtered_mols.append(ligand)
                
        num_filtered += len(filtered_mols)
        # print("NUM FILTERED MOLECULES: " + str(len(filtered_mols)))
        # print("MEAN FILTERED MOLECULES: " + str(np.mean([ligands[k][0] for k in filtered_mols])))
        sorted_filtered = sorted(filtered_mols, key=lambda k: ligands[k][0])
        best_10_filtered = []
        for i in sorted_filtered[:10]:
            best_10_filtered.append(ligands[i][0])
        
        c_filtered = cluster(sorted_filtered)
        c_filtered = sorted(c_filtered, key=lambda k: ligands[k][0])
        best_10_cluster_filtered = []
        for i in c_filtered[:10]:
            best_10_cluster_filtered.append(ligands[i][0])
        
        filter_mean += np.mean(best_10_cluster_filtered)
        # print("MEAN TOP 10 FILTERED MOLECULES: " + str(np.mean(best_10_filtered)))
        # print("MEAN TOP 10 FILTERED MOLECULES (CLUSTERED): " + str(np.mean(best_10_cluster_filtered)))
        
        score_array = np.array(score_list)
        nds = NonDominatedSorting().do(score_array, only_non_dominated_front=True)
        pareto_front = np.array(ligands_list)[nds]
        
        pareto_size += len(pareto_front)
        pareto_mean += np.mean(np.array(pareto_affins)[nds])
        pareto_qed_mean += np.mean(np.array(pareto_qed)[nds])
        pareto_sa_mean += np.mean(np.array(pareto_sa))
        # print("PARETO FRONT SIZE: " + str(len(pareto_front)))
        # print("PARETO FRONT MEAN: " + str(np.mean(pareto_affins)))
        # print("PARETO FRONT QED: " + str(np.mean(pareto_qed)))
        # print("PARETO FRONT SA: " + str(np.mean(pareto_sa)))
        hv = HV(ref_point=np.array([1.0, 1.0, 1.0]))
        vals = np.array(score_list)[nds]
        hv = hv(np.array(vals))
        
        hypervolume += hv
        # print("HYPERVOLUME: " + str(hv))
        # get_all_ligands()
        
        # output_mols.append({key: ligands[key]} for key in c[:10])
        m_weights = [Descriptors.MolWt(Chem.MolFromSmiles(ligand)) for ligand in sorted_data]
        print(m_weights)
        running_avg = []
        cumsum = 0.0
        for i in range(len(m_weights)):
            if i == 0:
                running_avg.append(None)  # no values before index 0
            else:
                cumsum += m_weights[i - 1]
                running_avg.append(cumsum / i)

        x = list(range(len(m_weights)))

        plt.figure()
        plt.plot(x, running_avg, marker='o')
        plt.xlabel("Index")
        plt.ylabel("Average of values before index")
        plt.title("Running Average (excluding current value)")
        plt.show()

    print("AVG TOP TEN (CLUSTERED): " + str(np.mean(avg_top_10)))
    print("STDEV TOP 10 (CLUSTERED): " + str(np.std(avg_top_10)))
    print("AVG QED (clustered): " + str(avg_top_10_qed / num_seeds))
    print("AVG SA (clustered): " + str(avg_top_10_sa / num_seeds))
    print("AVG MAX SIM TOP 10: " + str(avg_top_10_max_sim / num_seeds))
    print("NUM BETTER THAN THRESHOLD: " + str(better_than_threshold / num_seeds))
    print("UNIQUE MEAN: " + str(np.mean(unique_mean)))
    print("UNIQUE STDEV: " + str(np.std(unique_mean)))
    print("AVERAGE UNIQUE MAX SIM: " + str(avg_max_sim_filtered / num_seeds))
    print()
    
    print("NUM FILTERED MOLECULES: " + str(num_filtered / num_seeds))
    print("MEAN FILTERED MOLECULES: " + str(filter_mean / num_seeds))
    print()
    
    print("PARETO FRONT SIZE: " + str(pareto_size / num_seeds))
    print("PARETO FRONT MEAN: " + str(pareto_mean / num_seeds))
    print("PARETO FRONT QED: " + str(pareto_qed_mean / num_seeds))
    print("PARETO FRONT SA: " + str(pareto_sa_mean / num_seeds))
    print("HYPERVOLUME: " + str(hypervolume / num_seeds))
    print()
    print()
    return unique_mean
values1 = analyze_results("GPT-oss_c-met_boltz_tool_use_optimized", limit=False, llm_only=True, eval_sim=False)
# values2 = analyze_results("GPT-4_brd4_boltz", limit=False, llm_only=True, eval_sim=True)
# values1 = analyze_results("GPT-4_c-met_bindingdb_docking", limit=False, llm_only=True, eval_sim=False)
# values3 = analyze_results("GPT-4_brd4_bindingdb_docking", limit=False, llm_only=True, eval_sim=False)
# values2 = analyze_results("GPT-4_brd4_boltz", limit=False, llm_only=True, eval_sim=False)
# values1 = analyze_results("GPT-4_brd4_bindingdb", limit=False, llm_only=True, eval_sim=False)
# _, p = ttest_ind(values1, values2, alternative="less", equal_var=False)
# print(p)
# create_yaml("GPT-4_c-met_zinc")
# set_similarity()

# runs = [filename.replace(".yaml", "") for filename in os.listdir("multi_objective/results") if "custom_c-met" in filename]
# runs.append("GPT-4_c-met_zinc")
# runs.append("GPT-4_c-met_summary")
# runs.append("GPT-4_c-met_base")
# runs.append("GPT-4_brd4_bindingdb")
# print(runs)
# results = {}
# llm_only_res = []
# not_llm_only_res = []
# for run in runs:
#     llm_only = np.mean(analyze_results(run, bindingdb=True, llm_only=True))
#     not_llm_only = np.mean(analyze_results(run, bindingdb=True, llm_only=False))
#     results[run] = llm_only
#     llm_only_res.append(llm_only)
#     not_llm_only_res.append(not_llm_only)
# sorted_results = sorted(results, key=results.get)
# for result in sorted_results:
#     print(f"{result}: {str(results[result])}")
# print("\n\n")
# print("LLM ONLY: ")
# for i in llm_only_res:
#     print(i)
# print("NOT LLM ONLY: ")
# for i in not_llm_only_res:
#     print(i)
# print(np.corrcoef(llm_only_res, not_llm_only_res))

# arr = ['CN(C)c1ccc(C2C3=C(CCCC3=O)Nc3[nH]c(=O)[nH]c(=O)c32)cc1CCN1CCNCC1c1ccc(O)c(F)c1Br', 'CC(Oc1ccc(C#N)cc1F)C(=O)NNC(=O)c1cc(Cl)c(Cl)c(Cl)c1NC(=O)N1CCN(C)C(=O)NC2=C3C=NC(=O)N=C3C(=O)N2CCN1c1ccc(Cl)cc1', 'CC(C#N)N1CCN(CCc2cc(C3C4=C(CCCC4=O)Nc4[nH]c(=O)[nH]c(=O)c43)ccc2N(C)C)C1c1cccc(F)c1NC(=O)c1ccc(Cl)c(F)c1']
# arr = ['CNCSCNCOC(COCF)C1(SO)C(F)NCCN1C1(C([NH2+]Cc2ccc(O)cc2)C2CCNCC2)CCSC1', 'CCCSC1CC(Oc2ncc(C(N)=O)cc2NC)CCC1N(CO)C(=O)Nc1ccc(C(OCC)SCSNNC)cc1F', 'CNCNN(C)CCCNC(=O)CN(CF)c1c(NCF)cc(C2C3=C(CCCC3=O)Nc3[nH]c(=O)[nH]c(=O)c32)c(SCCO)c1N1CCOCC1']
# mols = []
# labels = []
# for i in arr:
#     mol = Chem.MolFromSmiles(i)
#     mols.append(mol)
#     labels.append("SMILES: " + i + "\nQED: "+str(round(QED.qed(mol), 2)))
# img = Draw.MolsToGridImage(mols=mols, molsPerRow=2, subImgSize=(400,400), legends=labels)
# img.save(f"/home/ubuntu/MOLLEO/single_objective/script_images/molleo.png")