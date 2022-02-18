from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem
import numpy as np
import torch.multiprocessing as mp
from retro_star.api import RSPlanner


class InternalDiversity():
    def distance(self, mol1, mol2, dtype="Tanimoto"):
        assert dtype in ["Tanimoto"]
        if dtype == "Tanimoto":
            sim = DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(mol1), Chem.RDKFingerprint(mol2))
            return 1 - sim
        else:
            raise NotImplementedError

    def get_diversity(self, mol_list, dtype="Tanimoto"):
        similarity = 0
        mol_list = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in mol_list] 
        for i in range(len(mol_list)):
            sims = DataStructs.BulkTanimotoSimilarity(mol_list[i], mol_list[:i])
            similarity += sum(sims)
        n = len(mol_list)
        n_pairs = n * (n - 1) / 2
        diversity = 1 - similarity / n_pairs
        return diversity


class Synthesisability():
    def __init__(self):
        self.planner = RSPlanner(
                gpu=-1,
                starting_molecules='./retro_star/dataset/my_before_subm_origin_dict.csv',
                use_value_fn=True,
                iterations=500,
                expansion_topk=100)# 50)

    def get_syn_rate(self, mol_list):
        assert type(mol_list) == list
        syn_flag = []
        for i, mol_sml in enumerate(mol_list):
            result = self.planner.plan(Chem.MolToSmiles(mol_sml))
            if result:
                syn_flag.append(result['succ'])
            else:
                syn_flag.append(False)
        return np.mean(syn_flag)

if __name__ == "__main__":
    pass

