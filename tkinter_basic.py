import tkinter as tk
import pdb
import os

from grammar_generation import random_produce

from PIL import Image, ImageTk
from rdkit import Chem
from rdkit.Chem import Draw
import pickle5 as pickle
from os import listdir

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--polymer_log",help="dir for polymer log of grammar chkpt")
parser.add_argument("--iso_log")
parser.add_argument("--acy_log")
parser.add_argument("--chain_extender_log")
args = parser.parse_args()

    
window = tk.Tk()
expr_name_dict = dict()
expr_name_dict['polymer_117motif'] = args.polymer_log # 'test_run'
# expr_name_dict['iso'] = 'grammar-log/log_iso'
# expr_name_dict['acrylates'] = 'grammar-log/log_acy'
# expr_name_dict['chain_extender'] = 'grammar-log/log_ce'

expr_names = list(expr_name_dict.keys())
generated_mols = dict()
for expr_name in expr_names:
    print('dealing with {}'.format(expr_name))
    ckpt_list = listdir(expr_name_dict[expr_name])
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
    with open('{}/{}'.format(expr_name_dict[expr_name], max_R_ckpt), 'rb') as fr:
        grammar = pickle.load(fr)
    for i in range(8):
        mol, _ = random_produce(grammar)
        if expr_name not in generated_mols.keys():
            generated_mols[expr_name] = [mol]
        else:
            generated_mols[expr_name].append(mol)

# TODO: loop over exps
exp = 'polymer_117motif' # 'iso', 'acrylates', 'chain_extender'
dirname = f"./demo/{exp}/"
greeting = tk.Label(text="Hello, Tkinter")
greeting.pack()
os.makedirs(dirname,exist_ok=True)

path = dirname + f"mol{i}.png"    
pic = Chem.Draw.MolsToImage(generated_mols[exp], molsPerRow=1, subImgSize=(200,200))
pic.save(path)
mol_pic = ImageTk.PhotoImage(Image.open(path))
lab = tk.Label(image=mol_pic)
labimage = mol_pic
lab.pack()
correct_mols = set()
def handle_click(i,correct_mols):
    correct_mols.add(i)
    print(f"{i} is correct")

for i in range(len(generated_mols[exp])):
    button = tk.Button(
        text=f"Correct {i}",
        width=25,
        height=5,
        bg="blue",
        fg="yellow",
        command=lambda idx=i:handle_click(idx,correct_mols)
    )    
    button.pack()
window.mainloop()

with open(dirname + f"user_input.txt","w+") as f:
    for mol in correct_mols:
        f.write(f"{mol}\n")