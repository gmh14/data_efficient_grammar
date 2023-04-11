from grammar_generation import random_produce

import pickle5 as pickle
from rdkit import Chem
import os

def generate_mols(num_mol, log_dir, save_path, img_size=100):
    ckpt_list = os.listdir(log_dir)
    max_R = 0
    max_R_ckpt = None
    for ckpt in ckpt_list:
        if 'grammar' in ckpt:
            curr_R = float(ckpt.split('_')[-1][:-4])
            print(ckpt)
            if curr_R > max_R:
                max_R = curr_R
                max_R_ckpt = ckpt

    print('loading {}'.format(max_R_ckpt))
    with open('{}/{}'.format(log_dir, max_R_ckpt), 'rb') as fr:
        grammar = pickle.load(fr)

    generated_mols = []
    for i in range(num_mol):
        mol, _ = random_produce(grammar)
        generated_mols.append(mol)

    mol_img_list = []
    mol_sml_list = []
    for i, mol in enumerate(generated_mols):
        local_path = save_path + f'mol{i}.svg'
        pic = Chem.Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(100,100), useSVG=True)
        with open(local_path, 'w') as f_handle:
            f_handle.write(pic)
        mol_img_list.append(local_path)
        mol_sml_list.append(Chem.MolToSmiles(mol))

    return mol_sml_list, mol_img_list
