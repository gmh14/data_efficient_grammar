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
            # print(ckpt)
            if curr_R > max_R:
                max_R = curr_R
                max_R_ckpt = ckpt

    print('loading {}'.format(max_R_ckpt))
    with open('{}/{}'.format(log_dir, max_R_ckpt), 'rb') as fr:
        grammar = pickle.load(fr)

    generated_mols = []
    all_seq_list = []
    for i in range(num_mol):
        mol, _, mol_gen_seq = random_produce(grammar, gen_seq=True)
        generated_mols.append(mol)
        all_seq_list.append(mol_gen_seq)

    mol_img_list = []
    mol_sml_list = []
    mol_gen_seq_list = []
    for i, mol in enumerate(generated_mols):
        for fname in os.listdir(save_path + f'mol{i}_gen/'):
            os.remove(save_path + f'mol{i}_gen/' + fname)
        local_path = save_path + f'mol{i}.svg'
        pic = Chem.Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(100,100), useSVG=True)
        with open(local_path, 'w') as f_handle:
            f_handle.write(pic)
        gen_seq = []
        for k, mol_gen in enumerate(all_seq_list[i]):
            dir_path = save_path + f'mol{i}_gen'
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            local_gen_path = save_path + f'mol{i}_gen/' + 'gen_{}.svg'.format(k)
            pic = Chem.Draw.MolsToGridImage([mol_gen], molsPerRow=1, subImgSize=(100,100), useSVG=True)
            with open(local_gen_path, 'w') as f_handle:
                f_handle.write(pic)
            gen_seq.append(local_gen_path)

        mol_img_list.append(local_path)
        mol_sml_list.append(Chem.MolToSmiles(mol))
        mol_gen_seq_list.append(gen_seq)
    
    smi_path = save_path + 'gen_mol/'
    os.makedirs(smi_path, exist_ok=True)
    for i, sml in enumerate(mol_sml_list):
        with open(smi_path + f"{i}.smi", 'w+') as f:
            f.write(f"{sml}\n")


    return mol_sml_list, mol_img_list, mol_gen_seq_list
