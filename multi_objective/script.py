import os
import sys
import numpy as np
import yaml
from similarity_clustering import cluster
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import QED
from rdkit.Chem import DataStructs
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV

from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
        
def analyze_results(run_name, limit=False, llm_only=True, eval_sim=False):
    print("RUN: " + run_name)
    
    llm_ligands = []
    num_errors = 0
    if llm_only:
        with open(f"multi_objective/logs/{run_name}.txt", 'r') as log_file:
            for line in log_file:
                if "1000/1000" in line:
                    break
                if "LLM-GENERATED:" in line:
                    ligand = line.split()[1].strip()
                    llm_ligands.append(ligand)
                if "NUM LLM ERRORS" in line:
                    num_errors += 1
   
    cmet = []
    ligands = {}
    with open(f"multi_objective/results/results_{run_name}.yaml", 'r') as file:
        data = yaml.safe_load(file)
        for ligand, values in data.items():
            if limit and int(values[1]) > 255:
                continue
            if int(values[1]) <= 120:
                cmet.append(ligand)
            else:
                if (llm_only is False or ligand in llm_ligands) and float(values[0])!=0:
                    affin = float(values[0])
                    mol = Chem.MolFromSmiles(ligand)
                    qed = QED.qed(mol)
                    sa = sascorer.calculateScore(mol)
                    ligands[ligand] = [affin, qed, sa]
                    
    sorted_ligands = sorted(ligands, key=lambda k: ligands[k][0])
    # print(cmet)
    print(len(sorted_ligands))
    best_10 = []
    for i in sorted_ligands[:10]:
        best_10.append(ligands[i][0])
    
    c = cluster(sorted_ligands)
    c = sorted(c, key=lambda k: ligands[k][0])
    best_10_cluster = []
    for i in c[:10]:
        best_10_cluster.append(ligands[i][0])
    print("AVG TOP TEN: " + str(np.mean(best_10)))
    print("AVG TOP TEN (CLUSTERED): " + str(np.mean(best_10_cluster)))
    print("BEST: " + str(min(best_10_cluster)))
    print("STDEV TOP 10 (CLUSTERED): " + str(np.std(best_10_cluster)))
    print("BEST 10 LIGANDS (CLUSTERED):")
    qed = []
    sim = []
    sa = []
    num_better_than_threshold = 0
    threshold = -11
    unique = []
    for idx, ligand in enumerate(c):
        if idx < 10:
            qed.append(ligands[ligand][1])
            sa.append(ligands[ligand][2])
            
            mol = Chem.MolFromSmiles(ligand)
            morgan = AllChem.GetMorganGenerator(radius=2, fpSize=512)
            fingerprint = morgan.GetFingerprint(mol)
            max_sim = 0
            sim_ligand = ""
            for cmet_ligand in cmet:
                cmet_mol = Chem.MolFromSmiles(cmet_ligand)
                cmet_fingerprint = morgan.GetFingerprint(cmet_mol)
                similarity = DataStructs.TanimotoSimilarity(fingerprint, cmet_fingerprint)
                
                if similarity > max_sim:
                    max_sim = similarity
                    sim_ligand = cmet_ligand
            sim.append(max_sim)
            if ligands[ligand][0] < threshold:
                num_better_than_threshold += 1
            if max_sim < 0.5:
                unique.append(ligands[ligand][0])
            # print(ligand)
    print("AVG QED (clustered): " + str(np.mean(qed)))
    print("AVG SA (clustered): " + str(np.mean(sa)))
    print("STDEV QED: " + str(np.std(qed)))
    print("AVG MAX SIM: " + str(np.mean(sim)))
    print("NUMBER OF LLM ERRORS: " + str(num_errors))
    print("NUM BETTER THAN THRESHOLD: " + str(num_better_than_threshold))
    print("UNIQUE GENERATION MEAN: " + str(np.mean(sorted(unique)[:10])))
    print("UNIQUE GENERATION STD: " + str(np.std(sorted(unique)[:10])))
    
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
        
        single_score.append(1 - ligands[ligand][1])
        if ligands[ligand][1] < 0.6:
            pass_filter = False
        
        single_score.append((ligands[ligand][2]-1)/9)
        if ligands[ligand][2] > 3.0:
            pass_filter = False
        
        if "bindingdb" in run_name or eval_sim: 
            morgan = AllChem.GetMorganGenerator(radius=2, fpSize=512)
            mol = Chem.MolFromSmiles(ligand)
            fingerprint = morgan.GetFingerprint(mol)
            max_sim = 0
            for cmet_ligand in cmet:
                cmet_mol = Chem.MolFromSmiles(cmet_ligand)
                cmet_fingerprint = morgan.GetFingerprint(cmet_mol)
                similarity = DataStructs.TanimotoSimilarity(fingerprint, cmet_fingerprint)
                max_sim = max(max_sim, similarity)
            # single_score.append(max_sim)
            # pareto_sim.append(max_sim)
            if max_sim > 0.3:
                pass_filter = False
                    
        if not eval_sim or max_sim < 0.3: 
            score_list.append(single_score)
            pareto_affins.append(ligands[ligand][0])
            pareto_qed.append(ligands[ligand][1])
            pareto_sa.append(ligands[ligand][2])

        if pass_filter:
            filtered_mols.append(ligand)
            
    print()
    print("NUM FILTERED MOLECULES: " + str(len(filtered_mols)))
    print("MEAN FILTERED MOLECULES: " + str(np.mean([ligands[k][0] for k in filtered_mols])))
    sorted_filtered = sorted(filtered_mols, key=lambda k: ligands[k][0])
    best_10_filtered = []
    for i in sorted_filtered[:10]:
        best_10_filtered.append(ligands[i][0])
    
    c_filtered = cluster(sorted_filtered)
    c_filtered = sorted(c_filtered, key=lambda k: ligands[k][0])
    best_10_cluster_filtered = []
    for i in c_filtered[:10]:
        best_10_cluster_filtered.append(ligands[i][0])
    print("MEAN TOP 10 FILTERED MOLECULES: " + str(np.mean(best_10_filtered)))
    print("MEAN TOP 10 FILTERED MOLECULES (CLUSTERED): " + str(np.mean(best_10_cluster_filtered)))
    
    score_array = np.array(score_list)
    nds = NonDominatedSorting().do(score_array, only_non_dominated_front=True)
    pareto_front = np.array(ligands_list)[nds]
    print()
    print("PARETO FRONT SIZE: " + str(len(pareto_front)))
    print("PARETO FRONT MEAN: " + str(np.mean(pareto_affins)))
    print("PARETO FRONT QED: " + str(np.mean(pareto_qed)))
    print("PARETO FRONT SA: " + str(np.mean(pareto_sa)))
    if pareto_sim: print("PARETO FRONT SIM: " + str(np.mean(pareto_sim)))
    if pareto_sim: 
        hv = HV(ref_point=np.array([1.0, 1.0, 1.0, 1.0]))
    else:
        hv = HV(ref_point=np.array([1.0, 1.0, 1.0]))
    vals = np.array(score_list)[nds]
    hv = hv(np.array(vals))
    print("HYPERVOLUME: " + str(hv))
    
    return best_10_cluster
# get_all_ligands()
values1 = analyze_results("GPT-4_c-met_modified_2_repeat", limit=False, llm_only=True, eval_sim=True)
# values2 = analyze_results("GPT-4_brd4_boltz", limit=False, llm_only=False)
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

# arr = ['CC(N=O)c1ccc2nnc(Cc3c(F)cc4ncccc4c3F)n2n1', 'CC(c1c(F)cc2ncc(-c3cnn(C)c3)cc2c1F)n1nnc2ncc(-c3ccc(C(N)=O)c(F)c3)nc21', 'CNC(=O)c1ccc(-c2ccc(=O)n(CCc3ccc4ncc(-c5cnn(CCO)c5)cc4c3)n2)cc1C(F)(F)F', 'CC(=NNC(N)=O)c1cnc2nnn(Cc3cc4cccnc4cc3F)c2n1', 'O=C(Nc1ccc(Oc2ccnc(NC(=O)N3CCC3)c2)c(F)c1)c1cn(CC2CC2)c(=O)n(-c2ccc(F)cc2)c1=O', 'O=C1NCCCCn2cc(cn2)-c2cnc3ccc(cc3c2)CCn2nc(ccc2=O)-c2ccc1c(C(F)(F)F)c2', 'CC(c1c(F)cc2ncc(-c3cnn(C)c3)cc2c1F)n1nnc2ncc(-c3cnn(CCO)c3)nc21', 'COc1ccc2c(OCc3nnc4c(F)cc(-c5cc(C)cs5)cn34)ccnc2c1', 'O=c1ccc(-n2ccc3ccc(F)cc32)nn1CCOc1ccnc2cc(OCCCN3CCOCC3)ccc12', 'COc1cnc2ccc([C@H](C)c3nnc4c(F)cc(-c5cn(C)cn5)cn34)cc2c1', 'CC(=NNC(N)=O)c1ccc2nnc([C@@H](C)c3ccc4ncccc4c3)n2n1', 'C[C@@H](Oc1cc(C(=O)Nc2ccc(C(=O)N3CCOCC3)cc2)nnc1N)c1c(Cl)ccc(F)c1Cl', 'Cn1cc(-c2cnc3nnc(C(F)(F)c4ccc5ncccc5c4)n3n2)cn1', 'Cc1c(F)ccc(Cl)c1[C@@H](C)Oc1cc(C(=O)Nc2ccc(C(=O)N3CCN(C)CC3)cc2)nnc1N', 'CNC(=O)c1ccccc1Nc1nc(Nc2ccc3c(c2)C(C)CN(C(C)=O)CC3)ncc1Cl', 'Cc1csc(-c2ccc3nnc(Cc4ccc5ncccc5c4)n3n2)c1', 'O=C(Nc1cc(OC=CCc2c(F)cc3ncccc3c2F)ccn1)C1CC1', 'CNC(=O)c1ccc(-c2ccc3nnc(COc4ccnc5cc(OC)ccc45)n3c2)s1', 'Nc1ncnc2[nH]cc(-c3cc(F)c(NC(=O)c4cn(CC5CCOCC5)cc(-c5ccc(F)cc5)c4=O)cc3F)c12', 'COc1cc2nccc(Oc3ccc(NC(=O)c4c(I)ccn(-c5ccc(F)cc5)c4=O)cc3F)c2cc1OC', 'C[C@@H](Oc1c[nH]c(=O)c(C(=O)NC2CCOCC2)c1)c1c(Cl)ccc(F)c1Cl', 'CCN1Cc2c(ccc(F)c2C#N)O[C@@H](C)CNC(=O)c2c(N)nn3ccc1nc23', 'Oc1ccc(-c2cnc3nnc(C(F)(F)c4ccc5ncccc5c4)n3n2)cc1', 'CNC(=O)c1ccc(-c2cnc3nnc(C(F)(F)c4ccc5ncccc5c4)n3n2)cc1OC', 'CCN1Cc2c(ccc(F)c2C#N)O[C@@H](C)CNC(=O)c2c(F)nn3ccc1nc23', 'Cn1cc(-c2cnc3ccc4ccc(N)cc4c(=O)c3c2)cn1', '[C-]#[N+]C1=C2COCCN2C2=C(C(=O)OC2)C1c1ccc2[nH]nc(C)c2c1', 'FCCn1cc(-c2cnc3nnn(Cc4ccc5ncccc5c4)c3n2)cn1', 'COc1cc2nccc(Oc3ccc(NC(=O)c4c(C#N)ccn(-c5ccc(F)cc5)c4=O)cc3F)c2cc1O', 'CNC(=O)c1ccc(-c2ccc3nnc(Cc4ccc(O)cc4)n3n2)cc1Cl', 'O=C1C=NNNN1CC1C=C(Cc2ccc(O)c(Cc3ccc4nccc(Cl)c4c3)c2)CO1', 'CCc1ccc2nccc(OC3=CC=C(F)N(OCCN4CCNCC4)C=C3c3ccncc3)c2c1', 'Cc1ccc2nnc(Sc3ccc4ncc(OCO)cc4c3)n2c1', 'CNc1ccc2ccc3ncc(-c4cnn(C)c4)cc3c(=O)c2c1', 'Cc1csc(-c2ccc3nnc(Cc4c(F)cc(O)cc4F)n3n2)c1', 'Cc1csc(-c2ccc3nnc(C(=O)NCNCCn4cnc5nc(-c6cn[nH]c6)ccc54)n3n2)c1', 'Nc1nccc(Oc2ccc(F)cc2I)c1C(=O)Nc1ccncc1', 'CC(c1c(F)cc2ncc(-c3cnn(CCO)c3)cc2c1F)n1nnc2ncc(-c3cnn(C)c3)nc21', 'C1=CSC(C2=Nn3c(nnc3Cc3ccc4ncccc4c3)C2)C1', 'CCN(C)CCNC(=O)c1cc(-c2cnc3nnc(C(F)(F)c4ccc5ncccc5c4)n3n2)ccc1O', 'CC(NC/N=C/c1cnn(C)c1)c1ccc2nnc(Cc3c(F)cc4ncccc4c3F)n2n1', 'CC(NC/N=C/c1cnn(C)c1)c1ccc2ncccc2c1', 'COc1cc2nccc(Oc3ccc(NC(=O)c4c(C#N)ccn(-c5ccc(F)cc5)c4=O)cc3C)c2cc1O', 'CCc1ccc2ncccc2c1Nc1ccc(OC)cc1C(N)=O', 'Cc1ccc2nnc(Sc3ccc4ncc(OCCO)cc4c3)n2c1', 'CCOc1cc2cc(N3CCN(C)C(Cc4ccc(F)cc4)C3)cnc2nn1', '[C-]#[N+]C1=C2COCCN2C2=C(C(=O)OC2)C1C1=CC=C(Cn2nc(-c3cnn(C)c3)ccc2=O)C1', 'Oc1ccc(Cc2nnc3ccc(-c4cccs4)nn23)cc1F', 'CC1=CSC2=CC(F)CNCCN1CCOc1ccc(cc1)Cc1nnc3ccc2nn13', 'CC(=O)Nc1ccc(C(F)(F)c2nnc3ncc(-c4ccc(O)cc4)nn23)cc1', 'CNc1snc(C(=O)Nc2ccc(F)cc2N2CCN(C(=O)c3cn(CC4CC4)c(=O)n(-c4ccc(F)cc4)c3=O)CC2)c1C(N)=O', 'CN1CCN(c2cnc3nnc(C#N)cc3c2)CC1Cc1ccc(F)cc1', 'CNC(=O)c1ccccc1Nc1ccnc(Nc2ccc(OC)c3nnc(CO)c(Cl)c23)n1', 'Fc1cc(-c2ccc3nnc(Cc4ccc(OC(F)(F)F)nc4)n3n2)cc(N2CCNCC2)c1', 'Cc1csc(-c2ccc3nnc(Cc4ccc(O)c(F)c4)n3n2)c1', 'Cc1ccc(Cl)c2c1NC(=O)C2=NC(=O)c1ccc(O)cc1', 'CNC(=O)c1ccc(C2C=CN(C(=O)c3cn(C)c(=O)n(-c4cc(F)ccc4F)c3=O)CC2)cc1Cl', 'CNC(=O)c1cc(OC2CCNCC2)ccc1Nc1nc(Nc2ccc3c(c2)C(C)CN(C(C)=O)CC3)ncc1F', 'C[C@@H](Oc1c[nH]c(=O)c(C(=O)Nn2cc(-c3ccc(Cl)cc3Cl)nn2)c1)c1c(Cl)ccc(F)c1Cl', 'CC1=NC(C)=C(C#N)C(c2ccc3[nH]nc(OCCOC(C)C)c3c2)C1C#N', 'O=CNCc1ccc(O)c(Cl)c1-c1cnc2ncc(Cc3cc(Cl)c4ncccc4c3)n2c1', 'O=C(Cc1ccc(O)cc1)NNC(=O)N1C=C(c2ccncc2)C(Cl)=CC=C1F', 'CCc1ccc2ncccc2c1C(=O)NCc1ccc(C(F)(F)F)cc1C#N', 'COc1ccc(Cl)c2c1NC(=O)C2=NC(=O)c1ccc(O)cc1F', 'Oc1ccc(Cc2nnc3ccc(C4CC=CS4)nn23)cc1', 'COc1ccc(-c2cscn2)c2c1NC(=O)C2=NC(=O)c1ccc(O)cc1F', 'Cc1csc2nccc(Nc3ccc(O)c(F)c3)c12', 'Cc1ccc(Cl)c2c1N=C(F)C2=NNC(=O)[C@@H](C)c1ccc(O)cc1', 'CNc1nccn1-c1cc(Cl)cc(Cl)c1OCCOc1ccc(F)cc1O', 'O=C(Nc1ccc(OC2=CC=NC3=CCCN2C(=O)N3)c(F)c1)c1cn(CC2CC2)c(=O)n(-c2ccc(F)cc2)c1=O', 'COc1ccc2c(OCc3nnc4c(F)cc(-c5ccsc5C)cn34)ccnc2c1', 'CNc1nccn1-c1cc(Cl)cc(Cl)c1OCCN1CCNCC1F', 'Nc1nccc(Oc2ccc(F)cc2I)c1C(=O)NCc1nnc2ncc(-c3ccc(O)cc3)nn12', 'COc1ccc2c(COc3nnc4ccc(C5CN(c6cnc7cc(Cl)c(F)c(C(F)(F)F)c7c6F)CCN5C)cn34)ccnc2c1', 'O=C(Nc1cc(Oc2cc(F)c(NC(=O)N3CCOCC3)cc2F)ccn1)C1CC1', 'O=c1[nH]c2ccccc2n(-c2ccc(N3CCN(Cc4ccc(C(F)(F)F)cc4)CC3)cc2)c1=O', 'CN1CCN(c2cnc3cc(C#N)cc(OCc4ccc(N)cc4)c3c2)CC1C=O', 'COc1ccncc1CNCNCCN1CCOCC1Oc1ccc(NC(=O)c2c(I)ccncc2=O)cc1F', 'COc1ccc(NC(=O)c2cn(CC3CC3)c(=O)n(-c3ccc(F)cc3)c2=O)c(F)c1', 'CCc1cc(Cc2nnc3ccsc3n2)ccc1O', 'Cc1cc(CC2CNC3=C(F)C=CNN32)cc2cccnc12', 'COc1ccn(CCc2ccc(O)c(F)c2)n1', 'CCOc1ccc2nnc(Cc3ccc(O)cc3)n2n1', 'COc1cc2nccc(OC(C)(F)F)c2cc1OC', 'O=C(Nc1ccc(Oc2ccnc(NC(=O)N3CCN(c4cnc5cc(C(F)(F)F)c(Cl)cc5c4F)CC3)c2)cc1)c1cnc2ccccc2[n+]1[O-]', 'COc1ccc(Cc2nnc3ccc(NC(=O)C4CC4)nn23)cn1', 'CN1CCN(c2cnc3cc(C#N)cc(OCc4ccc(N)cc4)c3c2)CC1', 'CN1CCN(C2=CN3C(=NN=C(C(C)(F)F)c4ccc(F)cc43)N=C2)CC1', 'Cc1csc(C2=NN=C3N=CC(c4ccc(O)cc4)=NN3C2)c1', 'FC1=C2NCC(Cc3cc(Cl)c4ncccc4c3)N2NC=C1', 'COc1ccc2c(NCc3ccc(F)cc3)nc3ncnc(S)c3c2c1I', 'CC(C)n1nnc2ncc(-c3ccc(C(N)=O)c(F)c3)nc21', 'CCc1ccc2ncccc2c1Nc1ccc(C)cc1', 'N#Cc1ccc2[nH]nc(C(=O)Nc3ccc(F)cc3)c2c1', 'O=C1C=NNNSc2ccc(O)c(F)c2C1', 'OC1=CN(OCCN2CCNCC2)C(F)=CC=C1Cl', 'CCc1ccc2nc(F)sc2c1-c1cnn[nH]1', 'N#Cc1ccn(-c2ccc(F)cc2)c(=O)c1C(=O)Nc1ccc(F)cc1', 'CCN1CCC(n2cc(-c3ccncc3)cn2)CC1', 'CCc1ccc2nccc(Oc3ccc(F)cc3)c2c1', 'COc1ccc(NC(=O)c2ncc(F)cc2I)cc1NCCN1CCOCC1', 'Oc1ccc(F)cc1', 'CN1CCN(c2cccnc2C#N)CC1c1ccc(F)cc1', 'CCc1ccc2ncccc2c1Nc1ccc(F)cc1', 'COc1ccc(-c2nccc(C#N)n2)cc1', 'CCNC(=O)c1ccc2ncc(OC)c(F)c2c1', 'N#Cc1cc(F)ccc1-c1cnc(NC2CCOCC2)nn1', 'CN1CCN(CCOc2cnc(-c3cc(F)cc(C#N)c3)nc2)CC1', 'O=C(NCN1CCOCC1)Nc1ccccc1', 'CCOc1ccc2nccc(Oc3ccc(F)cc3)c2c1', 'FC1=CC=C(Cl)C(c2ccncc2)=CN1OCCN1CCNCC1', 'CCN1CCC(O)(n2cc(-c3ccncc3)cn2)CC1', 'COc1ccc2nccc(Oc3ccc(F)cc3)c2c1', 'CCN1CCC(COc2cnc(-c3cccc(OC)c3)nc2)C(O)C1', 'NCOC1=COCC1', 'COC(C)n1nnc2nnn(C)c21', 'C1=NNCCO1', 'CNC(=O)c1ccc(-c2ccc(O)cc2)cc1Cl', 'CN1CCN(C(=O)Nc2ccc(Cl)cc2F)CC1', 'CN1CCN(c2cnc(Cl)c(C#N)c2F)CC1', 'CCc1ccc2ncccc2c1', 'COc1ccc(F)cc1-c1ccncc1F', 'FC1=CNC2=C(F)C=CNN2C1', 'Cn1cnc(N2CCCCC2)c1', 'CC1=C[SH]=NN=C1', 'CNC(=O)C(F)(F)c1ccc(OC)nc1', 'CCCN1CCN(C)CC1', 'CCN1CCC(n2cc3ncccc3n2)CC1', 'COC(C)=Cc1cccnc1', 'Clc1ccc2ncccc2c1', 'CCc1ccc2nccc(Cl)c2c1', 'CCc1ccc(N)cc1', 'CCCOC(C)C', 'CC(F)(F)C#N', 'CC1=CN(C)CC=C1', 'CCc1ccc2nc(F)sc2c1', 'Fc1ccc2ncccc2c1', 'CC(=O)Nc1cccnc1', 'CNNc1ccccc1C(=O)NC', 'COc1ccccc1C(=O)N(C)C']
# for i in arr:
#     mol = Chem.MolFromSmiles(i)
#     smiles = Chem.MolToSmiles(mol, canonical=True)
#     print(smiles)
# print(len(arr))