import traceback
import openai
import re
from rdkit import Chem
import main.molleo_multi_pareto.crossover as co, main.molleo_multi_pareto.mutate as mu
import random
from dotenv import load_dotenv
from openai import OpenAI
import os
import sys

from rdkit.Chem import QED
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


load_dotenv()
client = OpenAI(api_key=os.getenv("GPT_KEY"))
MINIMUM = 1e-10


def query_LLM(messages, model="gpt-4.1-mini", temperature=0.0):
    message = [{"role": "system", "content": "You are a helpful agent who can answer the question based on your molecule knowledge."}]
    message += messages

    params = {
        "model": model, #Please use your own open engine
        "max_completion_tokens": 4096,
        "messages": message
    }

    for retry in range(3):
        try:
            obj = client.chat.completions.create(**params)
            response = obj.choices[0].message.content
            message.append({"role": "assistant", "content": response})
            break
        except Exception as e:
            traceback.print_exc()
            print(f"{type(e).__name__} {e}")


    print("=>")
    return response

class GPT4:
    def __init__(self):
        self.task2description_mul = {
                'qed_c-met_sa': 'I have two molecules and their QED, SA (Synthetic Accessibility), and Binding Affinity against the kinase c-MET.\n',
                'qed_brd4_sa': 'I have two molecules and their QED, SA (Synthetic Accessibility), and Binding Affinity against BRD4.\n',
                '3': 'I have two molecules and their QED, SA (Synthetic Accessibility), JNK3 (biological activity against the kinase JNK3), GSK3$\beta$ (biological activity against Glycogen Synthase Kinase 3 Beta), DRD2 (biological activity against a biological target named the dopamine type 2 receptor (DRD2)) scores.\n\n',
                }
        self.task2objective_mul = {
                'qed_c-met_sa': '\n\nI want to maximize QED score, minimize affinity score, and minimize SA score. Please propose a new molecule better than the current molecules. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'qed_brd4_sa': '\n\nI want to maximize QED score, maximize affinity score, and minimize SA score. Please propose a new molecule better than the current molecules. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                '3': 'I want to maximize QED score, maximize JNK3 score, minimize GSK3$\beta$ score, minimize SA score and minimize DRD2 score. Please propose a new molecule better than the current molecules. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                }
        
        # self.task2description_mul_single = {
        #         'qed_c-met_sa': 'I have a molecule and its QED, SA (Synthetic Accessibility), and Binding Affinity against the kinase c-MET.\n',
        #         'qed_brd4_sa': 'I have a molecule and its QED, SA (Synthetic Accessibility), and Binding Affinity against BRD4.\n',
        #         '3': 'I have two molecules and their QED, SA (Synthetic Accessibility), JNK3 (biological activity against the kinase JNK3), GSK3$\beta$ (biological activity against Glycogen Synthase Kinase 3 Beta), DRD2 (biological activity against a biological target named the dopamine type 2 receptor (DRD2)) scores.\n\n',
        #         }
        # self.task2objective_mul_single = {
        #         'qed_c-met_sa': '\n\nI want to maximize QED score, minimize affinity score, and minimize SA score. Please propose a new molecule better than the current molecule. You can either make a mutation based on the given molecule or just propose a new molecule based on your knowledge.\n\n',
        #         'qed_brd4_sa': '\n\nI want to maximize QED score, maximize affinity score, and minimize SA score. Please propose a new molecule better than the current molecule. You can either make a mutation based on the given molecule or just propose a new molecule based on your knowledge.\n\n',
        #         '3': 'I want to maximize QED score, maximize JNK3 score, minimize GSK3$\beta$ score, minimize SA score and minimize DRD2 score. Please propose a new molecule better than the current molecules. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                # }
        self.requirements = """Your output should follow the format: {<<<Explaination>>>: $EXPLANATION, <<<Molecule>>>: \\box{$Molecule}}. Here are the requirements:\n1. $EXPLANATION should be your analysis.\n2. The $Molecule should be the smiles of your propsosed molecule.\n3. The molecule should be valid."""
        self.task=None
        self.error_count = 0

    def edit(self, mating_tuples, mutation_rate, min_evaluators, max_evaluators, boltz_cache, single_parent=False):
        task = self.task
        all_objectives = min_evaluators + max_evaluators
        obj_map = {"qed": "QED", "sa": "SA (Synthetic Accessibility)", "c-met": "Binding Affinity against the kinase c-MET", "brd4": "Binding Affinity against BRD4"}
        
        task_definition = "I have a molecule and its "
        for objective in all_objectives[:-1]:
            task_definition += obj_map[objective[0]]
            task_definition += ", "
        task_definition += "and "
        task_definition += obj_map[all_objectives[-1][0]]
        task_definition += ".\n"
        
        parent = []
        parent.append(random.choice(mating_tuples))
        parent.append(random.choice(mating_tuples))
        parent_smiles = [t[1] for t in parent]
        parent_scores = [t[0] for t in parent]
        print(parent_smiles)
        try:
            mol_tuple = ''
            for i in range(len(parent)):
                smiles = parent_smiles[i]
                tu = f'\n{str(i+1)}. {smiles}\n'
                for obj in all_objectives: 
                    tu += obj_map[obj[0]]
                    tu+= ": "
                    if obj[0] == "c-met" or obj[0] == "brd4":
                        score = boltz_cache[obj[0]][smiles]
                    else:    
                        mol = Chem.MolFromSmiles(smiles)
                        score = obj[1](mol)
                    tu += str(score)
                    tu += "\n"
                mol_tuple = mol_tuple + tu
                
            task_objective = "\nI want to "
            for min_obj in min_evaluators:
                if min_obj[0] == "c-met" or min_obj[0] == "brd4":
                    task_objective += "improve "
                else:
                    task_objective += "minimize "
                task_objective += obj_map[min_obj[0]]
                task_objective += ", "
            for max_obj in max_evaluators[:-1]:
                if max_obj[0] == "c-met" or max_obj[0] == "brd4":
                    task_objective += "worsen "
                else:
                    task_objective += "maximize "
                task_objective += obj_map[max_obj[0]]
                task_objective += ", "
            task_objective += "and maximize " if max_evaluators[-1][0] != "c-met" and max_evaluators[-1][0] != "brd4" else "and worsen "
            task_objective += obj_map[max_evaluators[-1][0]]
            task_objective += ". "
            task_objective += "Recall that a more negative binding affinity is better, and a more positive binding affinity is worse. Please propose a new molecule better than the current molecule. You can either make a mutation based on the given molecule or just propose a new molecule based on your knowledge.\n\n"

            prompt = task_definition + mol_tuple + task_objective + self.requirements
            
            print("Prompt: " + prompt, flush=True)
            messages = [{"role": "user", "content": prompt}]
            r = query_LLM(messages)
            print("Response: "  + r, flush=True)

            proposed_smiles = re.search(r'\\box\{(.*?)\}', r).group(1)
            proposed_smiles = sanitize_smiles(proposed_smiles)
            print(f"LLM-GENERATED: {proposed_smiles}", flush=True)
            print(proposed_smiles)
            assert proposed_smiles != None
            new_child = proposed_smiles

            return new_child
        except Exception as e:
            traceback.print_exc()
            print(f"{type(e).__name__} {e}")
            self.error_count += 1
            print("NUM LLM ERRORS: " + str(self.error_count), flush=True)
            mol1 = Chem.MolFromSmiles(parent_smiles[0])
            mol2 = Chem.MolFromSmiles(parent_smiles[1])
            new_child = co.crossover(mol1, mol2)
            smiles = None
            if new_child is not None:
                new_child = mu.mutate(new_child, mutation_rate)
            if new_child is not None: 
                smiles = Chem.MolToSmiles(new_child, canonical=True)
                smiles = sanitize_smiles(smiles)
                print(f"NON-LLM GENERATED: {smiles}")
            print(smiles)
            return smiles
            
            #single parent
            mol1 = Chem.MolFromSmiles(parent_smiles[0])
            new_child = mu.mutate(mol1, mutation_rate=1)
            if new_child is not None:
                smiles = Chem.MolToSmiles(new_child, canonical=True)
                smiles = sanitize_smiles(smiles)
                print(f"NON-LLM GENERATED: {smiles}")
                return smiles
            else:
                print(None)
                return None
                
    
def sanitize_smiles(smi):
    """
    Return a canonical smile representation of smi 

    Parameters
    ----------
    smi : str
        smile string to be canonicalized 

    Returns
    -------
    mol (rdkit.Chem.rdchem.Mol) : 
        RdKit mol object (None if invalid smile string smi)
    smi_canon (string)          : 
        Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): 
        True/False to indicate if conversion was  successful 
    """
    if smi == '':
        return None
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        smi_canon = Chem.MolToSmiles(mol, canonical=True)
        return smi_canon
    except:
        return None