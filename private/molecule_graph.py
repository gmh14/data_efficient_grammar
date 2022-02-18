from rdkit import Chem
from functools import partial
from fuseprop import extract_subgraph
from .hypergraph import mol_to_hg
from GCN.feature_extract import feature_extractor
from copy import deepcopy
import numpy as np

class MolGraph():
    def __init__(self, mol, is_subgraph=False, mapping_to_input_mol=None):
        if is_subgraph:
            assert mapping_to_input_mol is not None
        self.mol = mol
        self.is_subgraph = is_subgraph
        self.hypergraph = mol_to_hg(mol, kekulize=True, add_Hs=False)
        self.mapping_to_input_mol = mapping_to_input_mol

    def get_visit_status_edge(self, edge):
        return self.hypergraph.edge_attr(edge)['visited']

    def get_visit_status_node(self, node):
        return self.hypergraph.node_attr(node)['visited']

    def get_visit_status_with_idx(self, atom_idx):
        return self.get_visit_status_edge('e{}'.format(atom_idx))

    def set_visited(self, node_list, edge_list):
        for edge in edge_list:
            self.hypergraph.edge_attr(edge)['visited'] = True
        for node in node_list:
            self.hypergraph.node_attr(node)['visited'] = True

    def set_visit_status_with_idx(self, idx, visited):
        self.hypergraph.edge_attr('e{}'.format(idx))['visited'] = visited
    
    def set_NT_status_with_idx(self, idx, NT):
        self.hypergraph.edge_attr('e{}'.format(idx))['NT'] = NT

    def get_all_visit_status(self):
        return [self.get_visit_status_with_idx(i) for i in range(self.mol.GetNumAtoms())]

    def get_org_idx_in_input(self, idx):
        assert self.is_subgraph
        return self.mapping_to_input_mol.GetAtomWithIdx(idx).GetIntProp('org_idx')

    def get_org_node_in_input(self, node, subgraph):
        assert not self.is_subgraph
        adj_edges = list(subgraph.hypergraph.adj_edges(node))
        assert len(adj_edges) == 2
        org_edges = []
        for adj_edge_i in adj_edges:
            idx = int(adj_edge_i[1:])
            org_idx = subgraph.get_org_idx_in_input(idx)
            org_edges.append('e{}'.format(org_idx))
        org_node = list(self.hypergraph.nodes_in_edge(org_edges[0]).intersection(self.hypergraph.nodes_in_edge(org_edges[1])))
        try:
            assert len(org_node) == 1
        except:
            import pdb; pdb.set_trace()
        return org_node[0]
    
    def as_key(self):
        return MolKey(self.mol)

    def __eq__(self, another):
        # return hasattr(another, 'mol') and Chem.CanonSmiles(Chem.MolToSmiles(self.mol)) == Chem.CanonSmiles(Chem.MolToSmiles(another.mol))
        return hasattr(another, 'mol') and (Chem.MolToSmiles(self.mol)) == (Chem.MolToSmiles(another.mol))


class MolKey():
    def __init__(self, mol):
        if isinstance(mol, MolGraph):
            self.mol_graph = mol
            mol = mol.mol
        elif type(mol) == Chem.Mol:
            self.mol_graph = MolGraph(mol)
        else:
            raise TypeError
        # self.sml = Chem.CanonSmiles(Chem.MolToSmiles(mol))
        self.sml = Chem.MolToSmiles(mol)

    def __eq__(self, another):
        return hasattr(another, 'sml') and self.sml == another.sml

    def __hash__(self):
        return hash(self.sml)


class SubGraph(MolGraph):
    def __init__(self, mol, mapping_to_input_mol, subfrags):
        super(SubGraph, self).__init__(mol, is_subgraph=True, mapping_to_input_mol=mapping_to_input_mol)
        assert type(subfrags) == list
        '''
        subfrags: list, atom indices of two sub fragments
        bond: bond index of the connected bond
        '''
        self.subfrags = subfrags
        

class InputGraph(MolGraph):
    def __init__(self, mol, smiles, init_subgraphs, subgraphs_idx, GNN_model_path):
        '''
        init_subgraphs: a list of MolGraph
        subgraph_idx: a list of atom idx list for each subgraph
        '''
        super(InputGraph, self).__init__(mol)
        self.subgraphs = init_subgraphs
        self.subgraphs_idx = subgraphs_idx
        self.GNN_model_path = GNN_model_path
        self.smiles = smiles
        self.map_to_set = self.get_map_to_set()
        self.NT_atoms = set()
        self.rule_list = []
        self.rule_idx_list = []
        self.water_level = 0
        self.watershed_ext_nodes = {}

    def append_rule(self, rule, rule_idx):
        self.rule_list.append(rule)
        self.rule_idx_list.append(rule_idx)

    def get_map_to_set(self):
        map_to_set = dict()
        for i, subgraph in enumerate(self.subgraphs):
            key_subgraph = MolKey(subgraph)
            if key_subgraph not in map_to_set:
                map_to_set[key_subgraph] = list()
            map_to_set[key_subgraph].append(self.subgraphs_idx[i])
        return map_to_set

    def get_nodes_feature(self, id):
        # dimension order, N * dim_f
        self.feature_extractor = feature_extractor(self.GNN_model_path)
        nodes_feature = self.feature_extractor.extract(self.mol)
        return nodes_feature[id]
        
    def get_subg_feature_for_agent(self, subgraph):
        # Get feature vector for agent
        assert isinstance(subgraph, SubGraph)
        assert subgraph in self.subgraphs
        subfrags_feature = []
        nodes_feat = [self.get_nodes_feature(node_id).detach().cpu().numpy() for node_id in subgraph.subfrags]
        subfrags_feature = np.mean(nodes_feat, axis=0)
        return subfrags_feature # TODO could modify # should be an order-invariant function 

    def find_overlap(self, p_star_idx, subgraph_idx):
        union = []
        rst = len(set(subgraph_idx) & set(p_star_idx)) != 0
        if rst:
            union = list(set(subgraph_idx) | set(p_star_idx))
        return rst, union

    def set_water_level(self, node_list, edge_list, water_level):
        ext_node_list = []
        for edge in edge_list:
            self.hypergraph.edge_attr(edge)['water_level'] = water_level
            for _node in self.hypergraph.nodes_in_edge(edge):
                if _node not in node_list:
                    ext_node_list.append(_node)
        for node in node_list:
            self.hypergraph.node_attr(node)['water_level'] = water_level
        assert water_level not in self.watershed_ext_nodes.keys()
        self.watershed_ext_nodes[water_level] = ext_node_list

    def update_visit_status(self, visited_list):
        edge_list = ['e{}'.format(i) for i in visited_list]
        node_list = self.hypergraph.get_minimal_graph(edge_list)
        self.set_visited(node_list, edge_list)

    def update_NT_atoms(self, p_star_idx):
        _, p_subg_mapped, _ = extract_subgraph(self.smiles, p_star_idx)
        for idx, atom in enumerate(p_subg_mapped.GetAtoms()):
            org_idx = p_subg_mapped.GetAtomWithIdx(idx).GetIntProp('org_idx')
            if atom.GetAtomMapNum() == 1:
                self.NT_atoms.add(org_idx)
            else:
                if org_idx in self.NT_atoms:
                    self.NT_atoms.remove(org_idx)

    def update_watershed(self, p_star_idx):
        edge_list = ['e{}'.format(i) for i in p_star_idx]
        node_list = self.hypergraph.get_minimal_graph(edge_list)
        self.set_water_level(node_list, edge_list, self.water_level)
        self.water_level += 1

    def is_candidate_subgraph(self, subg):
        if subg.as_key() in self.map_to_set.keys():
            subgraphs_idx = self.map_to_set[subg.as_key()]
            subgraphs = [self.subgraphs[self.subgraphs_idx.index(idx_list)] for idx_list in subgraphs_idx]
            return True, subgraphs, subgraphs_idx
        return False, [], []

    def merge_selected_subgraphs(self, action_list):
        label_mapping = {} # label -> serial number
        label_mapping_inv = {} # serial number -> label
        label = 0
        selected_subg = []
        non_selected_subg = []
        p_star_list = []
        for i, action in enumerate(action_list):
            if action == 1:
                selected_subg.append((i, self.subgraphs[i], self.subgraphs_idx[i]))
            else:
                non_selected_subg.append((i, self.subgraphs[i], self.subgraphs_idx[i]))

        if len(selected_subg) == 1:
            return [selected_subg[0][1]]
        else:
            for i in range(len(selected_subg)):
                for j in range(i+1, len(selected_subg)):
                    selected_subg_idx_i = selected_subg[i][2]
                    selected_subg_idx_j = selected_subg[j][2]
                    selected_snum_i = selected_subg[i][0]
                    selected_snum_j = selected_subg[j][0]
                    if selected_snum_i not in label_mapping_inv.keys():
                        label_mapping_inv[selected_snum_i] = label
                    label += 1
                    rst, _ = self.find_overlap(selected_subg_idx_i, selected_subg_idx_j)
                    if rst:
                        label_i = label_mapping_inv[selected_snum_i]
                        label_mapping_inv[selected_snum_j] = label_i
                    else:
                        label_mapping_inv[selected_snum_j] = label
            for _key in label_mapping_inv.keys():
                _label = label_mapping_inv[_key]
                if _label not in label_mapping.keys():
                    label_mapping[_label] = [_key]
                else:
                    label_mapping[_label].append(_key)

            new_subgraphs = []
            new_subgraphs_idx = []
            for _key in label_mapping.keys():
                new_subgraph_idx = set()
                for snum in label_mapping[_key]:
                    new_subgraph_idx = new_subgraph_idx | set(self.subgraphs_idx[snum])
                new_subgraph_idx = list(new_subgraph_idx)
                subfrags = deepcopy(new_subgraph_idx)

                _, new_subgraph_mapped, _ = extract_subgraph(self.smiles, new_subgraph_idx)
                new_subgraph = SubGraph(new_subgraph_mapped, mapping_to_input_mol=new_subgraph_mapped, subfrags=subfrags)

                for idx, atom in enumerate(new_subgraph.mol.GetAtoms()):
                    org_idx = new_subgraph.get_org_idx_in_input(idx)
                    new_subgraph.set_visit_status_with_idx(idx, self.get_visit_status_with_idx(org_idx))
                    if org_idx in self.NT_atoms:
                        new_subgraph.set_NT_status_with_idx(idx, True)
                for node in new_subgraph.hypergraph.nodes:
                    org_node = self.get_org_node_in_input(node, new_subgraph)
                    new_subgraph.hypergraph.node_attr(node)['visited'] = self.hypergraph.node_attr(org_node)['visited']    
                new_subgraphs.append(new_subgraph)
                new_subgraphs_idx.append(new_subgraph_idx)
                p_star_list.append(new_subgraph)
            for non_selected_subg_i in non_selected_subg:
                new_subgraphs.append(non_selected_subg_i[1])
                new_subgraphs_idx.append(non_selected_subg_i[2])
            self.subgraphs = new_subgraphs
            self.subgraphs_idx = new_subgraphs_idx
            self.map_to_set = self.get_map_to_set()
            return p_star_list

    def update_subgraph(self, subg_idx):
        # Update visit_status and NT_atoms and watershed
        self.update_visit_status(subg_idx)
        self.update_NT_atoms(subg_idx)
        self.update_watershed(subg_idx)

        new_subgraphs = []
        new_subgraphs_idx = []
        for i, subg in enumerate(self.subgraphs):
            if self.subgraphs_idx[i] == subg_idx:
                continue
            else:
                rst, assemble = self.find_overlap(subg_idx, self.subgraphs_idx[i])
                if rst:
                    new_subgraph_idx = assemble
                    _, new_subgraph_mapped, _ = extract_subgraph(self.smiles, assemble)
                    subfrags = deepcopy(assemble)
                    new_subgraph = SubGraph(new_subgraph_mapped, mapping_to_input_mol=new_subgraph_mapped, subfrags=subfrags)
                    for idx, atom in enumerate(new_subgraph.mol.GetAtoms()):
                        org_idx = new_subgraph.get_org_idx_in_input(idx)
                        new_subgraph.set_visit_status_with_idx(idx, self.get_visit_status_with_idx(org_idx))
                        if org_idx in self.NT_atoms:
                            new_subgraph.set_NT_status_with_idx(idx, True)
                    for node in new_subgraph.hypergraph.nodes:
                        org_node = self.get_org_node_in_input(node, new_subgraph)
                        new_subgraph.hypergraph.node_attr(node)['visited'] = self.hypergraph.node_attr(org_node)['visited']    
                    new_subgraphs.append(new_subgraph)
                    new_subgraphs_idx.append(new_subgraph_idx)
                else:
                    new_subgraphs.append(subg)
                    new_subgraphs_idx.append(self.subgraphs_idx[i])

        self.subgraphs = new_subgraphs
        self.subgraphs_idx = new_subgraphs_idx
        self.map_to_set = self.get_map_to_set()

