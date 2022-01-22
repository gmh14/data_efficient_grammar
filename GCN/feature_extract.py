import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rdkit import Chem
from tqdm import tqdm
import numpy as np
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from GCN.model import GNN, GNN_feature
from GCN.loader import mol_to_graph_data_obj_simple


class feature_extractor():
    def __init__(self, pretrained_model_path):
        self.pretrained_model_path = pretrained_model_path

    def preprocessing(self, graph_mol):
        # rdkit_mol = AllChem.MolFromSmiles(graph_sml)
        # kekulize the molecule to distinguish the aromatic bonds
        data = mol_to_graph_data_obj_simple(Chem.MolFromSmiles(Chem.MolToSmiles(graph_mol)))
        # print("data.x, data.edge_index, data.edge_attr:", data.x, data.edge_index, data.edge_attr)
        # print(Chem.MolToSmiles(graph_mol))
        # import pdb; pdb.set_trace()
        return data

    def extract(self, graph_mol):
        model = GNN_feature(num_layer=5, emb_dim=300, num_tasks=1, JK='last', drop_ratio=0, graph_pooling='mean', gnn_type='gin')
        model.from_pretrained(self.pretrained_model_path)
        # model.cuda(device=0)
        model.eval()
        graph_data = self.preprocessing(graph_mol)
        # graph_data = graph_data.cuda(device=0)
        with torch.no_grad():
            node_features = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        del model
        return node_features

