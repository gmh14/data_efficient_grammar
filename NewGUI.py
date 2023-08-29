
from grammar_generation import random_produce
from UI_utils.local_file_picker  import local_file_picker
from UI_utils.grammar_gen import generate_mols
# from UI_utils.path_pages import *

from nicegui import ui, app
from nicegui.events import ValueChangeEventArguments
from nicegui.events import MouseEventArguments

from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import time
import os
from time import sleep
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--polymer_log", default="./grammar-log/log_117motifs", help="dir for polymer log of grammar chkpt")
parser.add_argument("--zoom", default=1., type=float, required=False, help="zoom in for each mol img")
parser.add_argument("--iso_log")
parser.add_argument("--acy_log")
parser.add_argument("--chain_extender_log")
args = parser.parse_args()

exp = 'polymer_117motif'
dirname = 'http://128.30.28.105:8000/'
local_dirname = 'demo/{}/'.format(exp)
os.makedirs(dirname,exist_ok=True)

app.add_static_files('/stl', 'demo')
mol_img_list = [local_dirname + f'mol{i}.svg' for i in range(8)]
# mol_img_list = [f'/stl/{i}_new.stl' for i in range(8)]
mol_sml_list = [None] * 8

shown_images = []

class Handler:
    def __init__(self, id):
        self.id = id
        self.r = 15
        self.down = False

    def set_origin_image(self, org_content):
        self.origin_content = org_content

    def mouse_handler(self, e: MouseEventArguments):
        color = 'SteelBlue' if e.type == 'mousedown' else 'Red'
        r = self.r
        print(e.type)
        if e.type == 'mousedown':
            self.x = e.image_x
            self.y = e.image_y
        elif e.type == 'mousemove':
            scale = np.linalg.norm([e.image_x - self.x, e.image_y - self.y]) * 0.1 + 1
            r = self.r * scale
            print(scale)
            print(r)
        else:
            scale = np.linalg.norm([e.image_x - self.x, e.image_y - self.y]) * 0.1 + 1
            r = self.r * scale
            print(scale)
            print(r)
        add_content = f'<circle cx="{self.x}" cy="{self.y}" r="{r}" fill="none" stroke="{color}" stroke-width="2" />'
        shown_images[self.id].content = self.origin_content + add_content

handlers = [Handler(i) for i in range(8)]

def new_samples_click():
    ui.notify(f'generating new samples...')
    mol_sml_list, mol_img_list, mol_gen_seq_list = generate_mols(8, args.polymer_log, local_dirname)
    for i in range(len(shown_images)):
        shown_images[i].source = '{}/{}?time={}'.format(dirname, mol_img_list[i], time.time())
        ui.update(shown_images[i])
        shown_images[i].update()
        radios[i].value = None



page_str = '''     
@ui.page('/path_page0')
def path_page0():
    print('test')
    path = local_dirname + f'mol0_gen/'
    files = os.listdir(path)
    print(files)
    num_rows = len(files) // 4 + 1
    for i in range(num_rows):
        with ui.row():
            for j in range(min(4, len(files) - i * 4)):
                with ui.card().style('width : {}px'.format(411-11)) as card:
                    ui.image(dirname + path + 'gen_{}.svg'.format(i * 4 + j)).style('width: {}%; height: {}%; object-fit: contain;'.format(111-1,111-1))
                    ui.label('Step {}'.format(i * 4 + j + 1))
    ui.button('RETURN', on_click=lambda: ui.open('/'))
'''
for i in range(8):    
    page_str_i = page_str.replace('0', str(i))
    if i == 6: print(page_str_i)
    exec(page_str_i)


                
radios = []
with ui.row():
    for i, (mol_sml, mol_img) in enumerate(zip(mol_sml_list[:4], mol_img_list[:4])):
        if mol_img.endswith('.svg'):
            with ui.card().style('width : 300px') as card:
                img_i = ui.interactive_image('{}/{}'.format(dirname, mol_img), on_mouse=handlers[i].mouse_handler, events=['mousedown', 'mousemove', 'mouseup'], cross=True).style('width: 100%; height: 100%; object-fit: contain;')
            handlers[i].set_origin_image(img_i.content)  
        else:
            with ui.scene() as scene:                
                scene.stl(mol_img)
            img_i = scene
              
        shown_images.append(img_i)
        radio_i = ui.radio({1: 'Correct', 0: 'Incorrect'}).props('inline')
        radios.append(radio_i)
        ui.button(f'Path {i}', on_click=lambda e: ui.open(eval('path_page{}'.format(e.sender.text[-1]))))

with ui.row():
    for i, (mol_sml, mol_img) in enumerate(zip(mol_sml_list[4:], mol_img_list[4:])):
        if mol_img.endswith('.svg'):
            with ui.card().style('width : 300px') as card:
                img_i = ui.interactive_image('{}/{}'.format(dirname, mol_img), on_mouse=handlers[i + 4].mouse_handler, events=['mousedown', 'mousemove', 'mouseup'], cross=True).style('width: 100%; height: 100%; object-fit: contain;')
            handlers[i + 4].set_origin_image(img_i.content)
        else:
            with ui.scene() as scene:
                scene.stl(mol_img)
            img_i = scene

        shown_images.append(img_i)
        radio_i = ui.radio({1: 'Correct', 0: 'Incorrect'}).props('inline')
        radios.append(radio_i)
        
        ui.button(f'Path {i+4}', on_click=lambda: ui.open(eval('path_page{}'.format(i+4))))

ui.button('New Samples', on_click=new_samples_click)
ui.run()
