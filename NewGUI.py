
from grammar_generation import random_produce
from UI_utils.local_file_picker  import local_file_picker
from UI_utils.grammar_gen import generate_mols

from nicegui import ui
from nicegui.events import ValueChangeEventArguments

from rdkit import Chem
from rdkit.Chem import Draw

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--polymer_log", required=True, default="./grammar-log/log_117motifs", help="dir for polymer log of grammar chkpt")
parser.add_argument("--zoom", default=1., type=float, required=False, help="zoom in for each mol img")
parser.add_argument("--iso_log")
parser.add_argument("--acy_log")
parser.add_argument("--chain_extender_log")
args = parser.parse_args()

exp = 'polymer_117motif'
dirname = 'http://localhost:8000/'
local_dirname = 'demo/{}/'.format(exp)
os.makedirs(dirname,exist_ok=True)

mol_sml_list, mol_img_list = generate_mols(6, args.polymer_log, local_dirname)

# async def pick_file() -> None:
#     result = await local_file_picker('~', multiple=True)
#     ui.notify(f'You chose {result}')

with ui.row():
    for mol_sml, mol_img in zip(mol_sml_list[:3], mol_img_list[:3]):
        with ui.card().style('width : 400px') as card:
            ui.image('{}/{}'.format(dirname, mol_img)).style('width: 100%; height: 100%; object-fit: contain;')
            with ui.card_section():
                ui.label(mol_sml)

with ui.row():
    for mol_sml, mol_img in zip(mol_sml_list[3:], mol_img_list[3:]):
        with ui.card().style('width : 400px') as card:
            ui.image('{}/{}'.format(dirname, mol_img)).style('width: 100%; height: 100%; object-fit: contain;')
            with ui.card_section():
                ui.label(mol_sml)

# ui.button('Choose file', on_click=pick_file).props('icon=folder')
ui.run()