from rdkit import Chem
from rdkit import RDLogger
from .symbol import TSymbol, BondSymbol
from copy import deepcopy
from typing import List, Dict, Tuple
from collections import Counter
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from private.utils import _node_match, _node_match_prod_rule, _edge_match, masked_softmax
from functools import partial
import numpy as np
import os


class Hypergraph(object):
    '''
    A class of a hypergraph.
    Each hyperedge can be ordered. For the ordered case,
    edges adjacent to the hyperedge node are labeled by their orders.

    Attributes
    ----------
    hg : nx.Graph
        a bipartite graph representation of a hypergraph
    edge_idx : int
        total number of hyperedges that exist so far
    '''
    def __init__(self):
        self.hg = nx.Graph()
        self.edge_idx = 0
        self.nodes = set([])
        self.num_nodes = 0
        self.edges = set([])
        self.num_edges = 0
        self.nodes_in_edge_dict = {}

    def __eq__(self, another):
        if self.num_nodes != another.num_nodes:
            return False
        if self.num_edges != another.num_edges:
            return False

        subhg_bond_symbol_counter \
                = Counter([self.node_attr(each_node)['symbol'] \
                for each_node in self.nodes])
        each_bond_symbol_counter \
                = Counter([another.node_attr(each_node)['symbol'] \
                for each_node in another.nodes])
        if subhg_bond_symbol_counter != each_bond_symbol_counter:
            return False

        subhg_atom_symbol_counter \
                = Counter([self.edge_attr(each_edge)['symbol'] \
                for each_edge in self.edges])
        each_atom_symbol_counter \
                = Counter([another.edge_attr(each_edge)['symbol'] \
                for each_edge in another.edges])
        if subhg_atom_symbol_counter != each_atom_symbol_counter:
            return False

        gm = GraphMatcher(self.hg,
                another.hg,
                partial(_node_match_prod_rule,
                    ignore_order=True),
                partial(_edge_match,
                    ignore_order=True))
        try:
            # next(gm.isomorphisms_iter())
            return gm.is_isomorphic()
        except StopIteration:
            return False

    def find_isomorphism_mapping(self, another, vis=False):
        assert self == another
        gm = GraphMatcher(self.hg,
                another.hg,
                partial(_node_match_prod_rule,
                    ignore_order=True),
                partial(_edge_match,
                    ignore_order=True))
        assert gm.is_isomorphic()
        # return gm.mapping
        return [mapping for mapping in gm.isomorphisms_iter()]

    def add_node(self, node: str, attr_dict=None):
        ''' add a node to hypergraph

        Parameters
        ----------
        node : str
            node name
        attr_dict : dict
            dictionary of node attributes
        '''
        self.hg.add_node(node, bipartite='node', attr_dict=attr_dict)
        if node not in self.nodes:
            self.num_nodes += 1
        self.nodes.add(node)

    def add_edge(self, node_list: List[str], attr_dict=None, edge_name=None):
        ''' add an edge consisting of nodes `node_list`

        Parameters
        ----------
        node_list : list 
            ordered list of nodes that consist the edge
        attr_dict : dict
            dictionary of edge attributes
        '''
        if edge_name is None:
            edge = 'e{}'.format(self.edge_idx)
        else:
            assert edge_name not in self.edges
            edge = edge_name
        self.hg.add_node(edge, bipartite='edge', attr_dict=attr_dict)
        if edge not in self.edges:
            self.num_edges += 1
        self.edges.add(edge)
        self.nodes_in_edge_dict[edge] = node_list
        if type(node_list) == list:
            for node_idx, each_node in enumerate(node_list):
                self.hg.add_edge(edge, each_node, order=node_idx)
                if each_node not in self.nodes:
                    self.num_nodes += 1
                self.nodes.add(each_node)

        elif type(node_list) == set:
            for each_node in node_list:
                self.hg.add_edge(edge, each_node, order=-1)
                if each_node not in self.nodes:
                    self.num_nodes += 1
                self.nodes.add(each_node)
        else:
            raise ValueError
        self.edge_idx += 1
        return edge

    def remove_node(self, node: str, remove_connected_edges=True):
        ''' remove a node

        Parameters
        ----------
        node : str
            node name
        remove_connected_edges : bool
            if True, remove edges that are adjacent to the node
        '''
        if remove_connected_edges:
            connected_edges = deepcopy(self.adj_edges(node))
            for each_edge in connected_edges:
                self.remove_edge(each_edge)
        self.hg.remove_node(node)
        self.num_nodes -= 1
        self.nodes.remove(node)

    def remove_nodes(self, node_iter, remove_connected_edges=True):
        ''' remove a set of nodes

        Parameters
        ----------
        node_iter : iterator of strings
            nodes to be removed
        remove_connected_edges : bool
            if True, remove edges that are adjacent to the node        
        '''
        for each_node in node_iter:
            self.remove_node(each_node, remove_connected_edges)

    def remove_edge(self, edge: str):
        ''' remove an edge

        Parameters
        ----------
        edge : str
            edge to be removed
        '''
        self.hg.remove_node(edge)
        self.edges.remove(edge)
        self.num_edges -= 1
        self.nodes_in_edge_dict.pop(edge)

    def remove_edges(self, edge_iter):
        ''' remove a set of edges

        Parameters
        ----------
        edge_iter : iterator of strings
            edges to be removed
        '''
        for each_edge in edge_iter:
            self.remove_edge(each_edge)

    def remove_edges_with_attr(self, edge_attr_dict):
        remove_edge_list = []
        for each_edge in self.edges:
            satisfy = True
            for each_key, each_val in edge_attr_dict.items():
                if not satisfy:
                    break
                try:
                    if self.edge_attr(each_edge)[each_key] != each_val:
                        satisfy = False
                except KeyError:
                    satisfy = False
            if satisfy:
                remove_edge_list.append(each_edge)
        self.remove_edges(remove_edge_list)

    def remove_subhg(self, subhg):
        ''' remove subhypergraph.
        all of the hyperedges are removed.
        each node of subhg is removed if its degree becomes 0 after removing hyperedges.

        Parameters
        ----------
        subhg : Hypergraph
        '''
        for each_edge in subhg.edges:
            self.remove_edge(each_edge)
        for each_node in subhg.nodes:
            if self.degree(each_node) == 0:
                self.remove_node(each_node)

    def nodes_in_edge(self, edge):
        ''' return an ordered list of nodes in a given edge.

        Parameters
        ----------
        edge : str
            edge whose nodes are returned

        Returns
        -------
        list or set
            ordered list or set of nodes that belong to the edge
        '''
        if edge.startswith('e'):
            return self.nodes_in_edge_dict[edge]
        else:
            adj_node_list = self.hg.adj[edge]
            adj_node_order_list = []
            adj_node_name_list = []
            for each_node in adj_node_list:
                adj_node_order_list.append(adj_node_list[each_node]['order'])
                adj_node_name_list.append(each_node)
            if adj_node_order_list == [-1] * len(adj_node_order_list):
                return set(adj_node_name_list)
            else:
                return [adj_node_name_list[each_idx] for each_idx
                        in np.argsort(adj_node_order_list)]

    def adj_edges(self, node):
        ''' return a dict of adjacent hyperedges

        Parameters
        ----------
        node : str

        Returns
        -------
        set
            set of edges that are adjacent to `node`
        '''
        return self.hg.adj[node]

    def adj_nodes(self, node):
        ''' return a set of adjacent nodes

        Parameters
        ----------
        node : str

        Returns
        -------
        set
            set of nodes that are adjacent to `node`
        '''
        node_set = set([])
        for each_adj_edge in self.adj_edges(node):
            node_set.update(set(self.nodes_in_edge(each_adj_edge)))
        node_set.discard(node)
        return node_set

    def has_edge(self, node_list, ignore_order=False):
        for each_edge in self.edges:
            if ignore_order:
                if set(self.nodes_in_edge(each_edge)) == set(node_list):
                    return each_edge
            else:
                if self.nodes_in_edge(each_edge) == node_list:
                    return each_edge
        return False

    def degree(self, node):
        return len(self.hg.adj[node])

    def degrees(self):
        return {each_node: self.degree(each_node) for each_node in self.nodes}

    def edge_degree(self, edge):
        return len(self.nodes_in_edge(edge))

    def edge_degrees(self):
        return {each_edge: self.edge_degree(each_edge) for each_edge in self.edges}

    def is_adj(self, node1, node2):
        return node1 in self.adj_nodes(node2)

    def get_minimal_graph(self, edge_list):
        candidate_node_list = set()
        for edge in edge_list:
            candidate_node_list.update(self.nodes_in_edge(edge))
        selected_nodes = []
        for candidate in candidate_node_list:
            cnt = 0
            for edge in edge_list:
                if candidate in self.nodes_in_edge(edge):
                    cnt += 1
            if cnt > 1:
                selected_nodes.append(candidate)
        return selected_nodes

    def adj_subhg(self, node, ident_node_dict=None):
        """ return a subhypergraph consisting of a set of nodes and hyperedges adjacent to `node`.
        if an adjacent node has a self-loop hyperedge, it will be also added to the subhypergraph.

        Parameters
        ----------
        node : str
        ident_node_dict : dict
            dict containing identical nodes. see `get_identical_node_dict` for more details

        Returns
        -------
        subhg : Hypergraph
        """
        if ident_node_dict is None:
            ident_node_dict = self.get_identical_node_dict()
        adj_node_set = set(ident_node_dict[node])
        adj_edge_set = set([])
        for each_node in ident_node_dict[node]:
            adj_edge_set.update(set(self.adj_edges(each_node)))
        fixed_adj_edge_set = deepcopy(adj_edge_set)
        for each_edge in fixed_adj_edge_set:
            other_nodes = self.nodes_in_edge(each_edge)
            adj_node_set.update(other_nodes)

            # if the adjacent node has self-loop edge, it will be appended to adj_edge_list.
            for each_node in other_nodes:
                for other_edge in set(self.adj_edges(each_node)) - set([each_edge]):
                    if len(set(self.nodes_in_edge(other_edge)) \
                           - set(self.nodes_in_edge(each_edge))) == 0:
                        adj_edge_set.update(set([other_edge]))
        subhg = Hypergraph()
        for each_node in adj_node_set:
            subhg.add_node(each_node, attr_dict=self.node_attr(each_node))
        for each_edge in adj_edge_set:
            subhg.add_edge(self.nodes_in_edge(each_edge),
                           attr_dict=self.edge_attr(each_edge),
                           edge_name=each_edge)
        subhg.edge_idx = self.edge_idx
        return subhg

    def get_subhg(self, node_list, edge_list, ident_node_dict=None):
        """ return a subhypergraph consisting of a set of nodes and hyperedges adjacent to `node`.
        if an adjacent node has a self-loop hyperedge, it will be also added to the subhypergraph.

        Parameters
        ----------
        node : str
        ident_node_dict : dict
            dict containing identical nodes. see `get_identical_node_dict` for more details

        Returns
        -------
        subhg : Hypergraph
        """
        if ident_node_dict is None:
            ident_node_dict = self.get_identical_node_dict()
        adj_node_set = set([])
        for each_node in node_list:
            adj_node_set.update(set(ident_node_dict[each_node]))
        adj_edge_set = set(edge_list)

        subhg = Hypergraph()
        for each_node in adj_node_set:
            subhg.add_node(each_node,
                           attr_dict=deepcopy(self.node_attr(each_node)))
        for each_edge in adj_edge_set:
            subhg.add_edge(self.nodes_in_edge(each_edge),
                           attr_dict=deepcopy(self.edge_attr(each_edge)),
                           edge_name=each_edge)
        subhg.edge_idx = self.edge_idx
        return subhg

    def copy(self):
        ''' return a copy of the object
        
        Returns
        -------
        Hypergraph
        '''
        return deepcopy(self)

    def node_attr(self, node):
        return self.hg.nodes[node]['attr_dict']

    def edge_attr(self, edge):
        return self.hg.nodes[edge]['attr_dict']

    def set_node_attr(self, node, attr_dict):
        for each_key, each_val in attr_dict.items():
            self.hg.nodes[node]['attr_dict'][each_key] = each_val

    def set_edge_attr(self, edge, attr_dict):
        for each_key, each_val in attr_dict.items():
            self.hg.nodes[edge]['attr_dict'][each_key] = each_val

    def get_identical_node_dict(self):
        ''' get identical nodes
        nodes are identical if they share the same set of adjacent edges.
        
        Returns
        -------
        ident_node_dict : dict
            ident_node_dict[node] returns a list of nodes that are identical to `node`.
        '''
        ident_node_dict = {}
        for each_node in self.nodes:
            ident_node_list = []
            for each_other_node in self.nodes:
                if each_other_node == each_node:
                    ident_node_list.append(each_other_node)
                elif self.adj_edges(each_node) == self.adj_edges(each_other_node) \
                   and len(self.adj_edges(each_node)) != 0:
                    ident_node_list.append(each_other_node)
            ident_node_dict[each_node] = ident_node_list
        return ident_node_dict
    '''
        ident_node_dict = {}
        for each_node in self.nodes:
            ident_node_dict[each_node] = [each_node]
        return ident_node_dict
    '''

    def get_leaf_edge(self):
        ''' get an edge that is incident only to one edge

        Returns
        -------
        if exists, return a leaf edge. otherwise, return None.
        '''
        for each_edge in self.edges:
            if len(self.adj_nodes(each_edge)) == 1:
                if 'tmp' not in self.edge_attr(each_edge):
                    return each_edge
        return None

    def get_nontmp_edge(self):
        for each_edge in self.edges:
            if 'tmp' not in self.edge_attr(each_edge):
                return each_edge
        return None

    def is_subhg(self, hg):
        ''' return whether this hypergraph is a subhypergraph of `hg`

        Returns
        -------
        True if self \in hg,
        False otherwise.
        '''
        for each_node in self.nodes:
            if each_node not in hg.nodes:
                return False
        for each_edge in self.edges:
            if each_edge not in hg.edges:
                return False
        return True

    def in_cycle(self, node, visited=None, parent='', root_node='') -> bool:
        ''' if `node` is in a cycle, then return True. otherwise, False.

        Parameters
        ----------
        node : str
            node in a hypergraph
        visited : list
            list of visited nodes, used for recursion
        parent : str
            parent node, used to eliminate a cycle consisting of two nodes and one edge.

        Returns
        -------
        bool
        '''
        if visited is None:
            visited = []
        if parent == '':
            visited = []
        if root_node == '':
            root_node = node
        visited.append(node)
        for each_adj_node in self.adj_nodes(node):
            if each_adj_node not in visited:
                if self.in_cycle(each_adj_node, visited, node, root_node):
                    return True
            elif each_adj_node != parent and each_adj_node == root_node:
                return True
        return False

    def get_all_NT_edges(self):
        NT_edges = []
        for edge in self.edges:
            edge_hg = Hypergraph()
            if not self.edge_attr(edge)['terminal']:
                node_list = list(self.nodes_in_edge(edge))
                for node in node_list:
                    edge_hg.add_node(node, deepcopy(self.node_attr(node)))
                edge_hg.add_edge(node_list, attr_dict=deepcopy(self.edge_attr(edge)), edge_name=edge)
                NT_edges.append(edge_hg)
        return NT_edges

    def draw_rule(self, lhs=False, file_path=None, with_edge_name=False):
        ''' draw hypergraph
        '''
        # plot nodes if lhs
        with_node = lhs

        import graphviz
        G = graphviz.Graph(format='png')
        for each_node in self.nodes:
            if 'ext_id' in self.node_attr(each_node):
                G.node(each_node, label='{}'.format(self.node_attr(each_node)['ext_id']), #'',
                       shape='circle', width='0.1', height='0.1', style='filled',
                       fillcolor='black', fontcolor='white')
            else:
                if with_node:
                    G.node(each_node, label='',
                           shape='circle', width='0.1', height='0.1', style='filled',
                           fillcolor='gray')
        edge_list = []
        for each_edge in self.edges:
            if self.edge_attr(each_edge).get('terminal', False):
                G.node(each_edge,
                       label=self.edge_attr(each_edge)['symbol'].symbol if not with_edge_name \
                       else self.edge_attr(each_edge)['symbol'].symbol + ', ' + each_edge,
                       fontcolor='black', shape='square')
            elif self.edge_attr(each_edge).get('tmp', False):
                G.node(each_edge, label='tmp' if not with_edge_name else 'tmp, ' + each_edge,
                       fontcolor='black', shape='square')
            else:
                G.node(each_edge,
                       label='{}*'.format(self.edge_attr(each_edge)['symbol'].symbol if not with_edge_name \
                       else self.edge_attr(each_edge)['symbol'].symbol + ', ' + each_edge),
                       fontcolor='black', shape='square', style='filled')
            # if with_node:
                # for each_node in self.nodes_in_edge(each_edge):
                    # G.edge(each_edge, each_node)
            # else:
            for each_node in self.nodes_in_edge(each_edge):
                if 'ext_id' in self.node_attr(each_node)\
                   and set([each_node, each_edge]) not in edge_list:
                    num_bond = 0
                    if self.node_attr(each_node)['symbol'].bond_type in [1, 2, 3]:
                        num_bond += self.node_attr(each_node)['symbol'].bond_type
                    elif self.node_attr(each_node)['symbol'].bond_type in [12]:
                        num_bond += 1
                    else:
                        raise NotImplementedError('unsupported bond type')
                    for _ in range(num_bond):
                        G.edge(each_edge, each_node)
                    edge_list.append(set([each_node, each_edge]))
            for each_other_edge in self.adj_nodes(each_edge):
                if set([each_edge, each_other_edge]) not in edge_list:
                    num_bond = 0
                    common_node_set = set(self.nodes_in_edge(each_edge))\
                                      .intersection(set(self.nodes_in_edge(each_other_edge)))
                    # Skip those edges that share nodes with external label
                    check_ext = ['ext_id' in self.node_attr(c_node) for c_node in common_node_set]
                    if not all(check_ext):
                        for each_node in common_node_set:
                            if self.node_attr(each_node)['symbol'].bond_type in [1, 2, 3]:
                                num_bond += self.node_attr(each_node)['symbol'].bond_type
                            elif self.node_attr(each_node)['symbol'].bond_type in [12]:
                                num_bond += 1
                            else:
                                raise NotImplementedError('unsupported bond type')
                        for _ in range(num_bond):
                            G.edge(each_edge, each_other_edge)
                        edge_list.append(set([each_edge, each_other_edge]))
        if file_path is not None:
            G.render(file_path, cleanup=True)
        return G
    
    def draw(self, file_path=None, with_node=False, with_edge_name=False, with_ext=True):
        ''' draw hypergraph
        '''
        import graphviz
        G = graphviz.Graph(format='png')
        for each_node in self.nodes:
            if 'ext_id' in self.node_attr(each_node) and with_ext:
                G.node(each_node, label='',
                       shape='circle', width='0.1', height='0.1', style='filled',
                       fillcolor='black')
            else:
                if with_node:
                    G.node(each_node, label='',
                           shape='circle', width='0.1', height='0.1', style='filled',
                           fillcolor='gray')
        edge_list = []
        for each_edge in self.edges:
            if self.edge_attr(each_edge).get('terminal', False):
                G.node(each_edge,
                       label=self.edge_attr(each_edge)['symbol'].symbol if not with_edge_name \
                       else self.edge_attr(each_edge)['symbol'].symbol + ', ' + each_edge,
                       fontcolor='black', shape='square')
            elif self.edge_attr(each_edge).get('tmp', False):
                G.node(each_edge, label='tmp' if not with_edge_name else 'tmp, ' + each_edge,
                       fontcolor='black', shape='square')
            else:
                G.node(each_edge,
                       label='{}*'.format(self.edge_attr(each_edge)['symbol'].symbol if not with_edge_name \
                       else self.edge_attr(each_edge)['symbol'].symbol + ', ' + each_edge),
                       fontcolor='black', shape='square', style='filled')
            if with_node:
                for each_node in self.nodes_in_edge(each_edge):
                    G.edge(each_edge, each_node)
            else:
                for each_node in self.nodes_in_edge(each_edge):
                    if 'ext_id' in self.node_attr(each_node)\
                       and set([each_node, each_edge]) not in edge_list and with_ext:
                        G.edge(each_edge, each_node)
                        edge_list.append(set([each_node, each_edge]))
                for each_other_edge in self.adj_nodes(each_edge):
                    if set([each_edge, each_other_edge]) not in edge_list:
                        num_bond = 0
                        common_node_set = set(self.nodes_in_edge(each_edge))\
                                          .intersection(set(self.nodes_in_edge(each_other_edge)))
                        for each_node in common_node_set:
                            if self.node_attr(each_node)['symbol'].bond_type in [1, 2, 3]:
                                num_bond += self.node_attr(each_node)['symbol'].bond_type
                            elif self.node_attr(each_node)['symbol'].bond_type in [12]:
                                num_bond += 1
                            else:
                                raise NotImplementedError('unsupported bond type')
                        for _ in range(num_bond):
                            G.edge(each_edge, each_other_edge)
                        edge_list.append(set([each_edge, each_other_edge]))
        if file_path is not None:
            G.render(file_path, cleanup=True)
            #os.remove(file_path)
        return G

    def is_dividable(self, node):
        _hg = deepcopy(self.hg)
        _hg.remove_node(node)
        return (not nx.is_connected(_hg))

    def divide(self, node):
        subhg_list = []

        hg_wo_node = deepcopy(self)
        hg_wo_node.remove_node(node, remove_connected_edges=False)
        connected_components = nx.connected_components(hg_wo_node.hg)
        for each_component in connected_components:
            node_list = [node]
            edge_list = []
            node_list.extend([each_node for each_node in each_component
                              if each_node.startswith('bond_')])
            edge_list.extend([each_edge for each_edge in each_component
                              if each_edge.startswith('e')])
            subhg_list.append(self.get_subhg(node_list, edge_list))
            #subhg_list[-1].set_node_attr(node, {'divided': True})
        return subhg_list

def mol_to_bipartite(mol, kekulize):
    """
    get a bipartite representation of a molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        molecule object

    Returns
    -------
    nx.Graph
        a bipartite graph representing which bond is connected to which atoms.
    """
    try:
        mol = standardize_stereo(mol)
    except KeyError:
        print(Chem.MolToSmiles(mol))
        raise KeyError
        
    # if kekulize:
        # Chem.Kekulize(mol)

    bipartite_g = nx.Graph()
    for each_atom in mol.GetAtoms():
        bipartite_g.add_node(f"atom_{each_atom.GetIdx()}",
                             atom_attr=atom_attr(each_atom, kekulize, terminal=(each_atom.GetAtomMapNum()!=1)))

    for each_bond in mol.GetBonds():
        bond_idx = each_bond.GetIdx()
        bipartite_g.add_node(
            f"bond_{bond_idx}",
            bond_attr=bond_attr(each_bond, kekulize))
        bipartite_g.add_edge(
            f"atom_{each_bond.GetBeginAtomIdx()}",
            f"bond_{bond_idx}")
        bipartite_g.add_edge(
            f"atom_{each_bond.GetEndAtomIdx()}",
            f"bond_{bond_idx}")
    return bipartite_g


def mol_to_hg(mol, kekulize, add_Hs):
    """
    get a bipartite representation of a molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        molecule object
    kekulize : bool
        kekulize or not
    add_Hs : bool
        add implicit hydrogens to the molecule or not.

    Returns
    -------
    Hypergraph
    """
    mol_org = mol
    if add_Hs:
        mol = Chem.AddHs(mol)

    # if kekulize:
        # Chem.Kekulize(mol)


    bipartite_g = mol_to_bipartite(mol, kekulize)
    hg = Hypergraph()
    for each_atom in [each_node for each_node in bipartite_g.nodes()
                      if each_node.startswith('atom_')]:
        node_set = set([])
        for each_bond in bipartite_g.adj[each_atom]:
            hg.add_node(each_bond,
                        attr_dict=bipartite_g.nodes[each_bond]['bond_attr'])
            node_set.add(each_bond)
            hg.add_node
        hg.add_edge(node_set,
                    attr_dict=bipartite_g.nodes[each_atom]['atom_attr'])
    return hg


def hg_to_mol(hg, verbose=False):
    """ convert a hypergraph into Mol object

    Parameters
    ----------
    hg : Hypergraph

    Returns
    -------
    mol : Chem.RWMol
    """
    mol = Chem.RWMol()
    atom_dict = {}
    bond_set = set([])
    for each_edge in hg.edges:
        atom = Chem.Atom(hg.edge_attr(each_edge)['symbol'].symbol)
        atom.SetNumExplicitHs(hg.edge_attr(each_edge)['symbol'].num_explicit_Hs)
        atom.SetFormalCharge(hg.edge_attr(each_edge)['symbol'].formal_charge)
        atom.SetChiralTag(
            Chem.rdchem.ChiralType.values[
                hg.edge_attr(each_edge)['symbol'].chirality])
        atom_idx = mol.AddAtom(atom)
        atom_dict[each_edge] = atom_idx

    for each_node in hg.nodes:
        edge_1, edge_2 = hg.adj_edges(each_node)
        if edge_1+edge_2 not in bond_set:
            if hg.node_attr(each_node)['symbol'].bond_type <= 3:
                num_bond = hg.node_attr(each_node)['symbol'].bond_type
            elif hg.node_attr(each_node)['symbol'].bond_type == 12:
                num_bond = 1
            else:
                raise ValueError(f'too many bonds; {hg.node_attr(each_node)["bond_symbol"].bond_type}')
            _ = mol.AddBond(atom_dict[edge_1],
                            atom_dict[edge_2],
                            order=Chem.rdchem.BondType.values[num_bond])
            bond_idx = mol.GetBondBetweenAtoms(atom_dict[edge_1], atom_dict[edge_2]).GetIdx()

            # stereo
            mol.GetBondWithIdx(bond_idx).SetStereo(
                Chem.rdchem.BondStereo.values[hg.node_attr(each_node)['symbol'].stereo])
            bond_set.update([edge_1+edge_2])
            bond_set.update([edge_2+edge_1])
    mol.UpdatePropertyCache()
    mol = mol.GetMol()
    not_stereo_mol = deepcopy(mol)
    if Chem.MolFromSmiles(Chem.MolToSmiles(not_stereo_mol)) is None:
        raise RuntimeError('no valid molecule was obtained.')
    try:
        mol = set_stereo(mol)
        is_stereo = True
    except:
        import traceback
        traceback.print_exc()
        is_stereo = False
    mol_tmp = deepcopy(mol)
    Chem.SetAromaticity(mol_tmp)
    if Chem.MolFromSmiles(Chem.MolToSmiles(mol_tmp)) is not None:
        mol = mol_tmp
    else:
        if Chem.MolFromSmiles(Chem.MolToSmiles(mol)) is None:
            mol = not_stereo_mol
    mol.UpdatePropertyCache()
    if verbose:
        return mol, is_stereo
    else:
        return mol


def atom_attr(atom, kekulize, terminal):
    """
    get atom's attributes

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
    kekulize : bool
        kekulize or not

    Returns
    -------
    atom_attr : dict
        "is_aromatic" : bool
            the atom is aromatic or not.
        "smarts" : str
            SMARTS representation of the atom.
    """
    if kekulize:
        return {'terminal': terminal,
                'is_in_ring': atom.IsInRing(),
                'visited': False, 
                'NT': False, 
                'symbol': TSymbol(degree=0,
                                  is_aromatic=False,
                                  symbol=atom.GetSymbol(),
                                  num_explicit_Hs=atom.GetNumExplicitHs(),
                                  formal_charge=atom.GetFormalCharge(),
                                  chirality=atom.GetChiralTag().real
                )}
    else:
        return {'terminal': terminal,
                'is_in_ring': atom.IsInRing(),
                'visited': False, 
                'NT': False, 
                'symbol': TSymbol(degree=0,
                                  is_aromatic=atom.GetIsAromatic(),
                                  symbol=atom.GetSymbol(),
                                  num_explicit_Hs=atom.GetNumExplicitHs(),
                                  formal_charge=atom.GetFormalCharge(),
                                  chirality=atom.GetChiralTag().real
                )}

def bond_attr(bond, kekulize):
    """
    get atom's attributes

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
    kekulize : bool
        kekulize or not

    Returns
    -------
    bond_attr : dict
        "bond_type" : int
        {0: rdkit.Chem.rdchem.BondType.UNSPECIFIED,
         1: rdkit.Chem.rdchem.BondType.SINGLE,
         2: rdkit.Chem.rdchem.BondType.DOUBLE,
         3: rdkit.Chem.rdchem.BondType.TRIPLE,
         4: rdkit.Chem.rdchem.BondType.QUADRUPLE,
         5: rdkit.Chem.rdchem.BondType.QUINTUPLE,
         6: rdkit.Chem.rdchem.BondType.HEXTUPLE,
         7: rdkit.Chem.rdchem.BondType.ONEANDAHALF,
         8: rdkit.Chem.rdchem.BondType.TWOANDAHALF,
         9: rdkit.Chem.rdchem.BondType.THREEANDAHALF,
         10: rdkit.Chem.rdchem.BondType.FOURANDAHALF,
         11: rdkit.Chem.rdchem.BondType.FIVEANDAHALF,
         12: rdkit.Chem.rdchem.BondType.AROMATIC,
         13: rdkit.Chem.rdchem.BondType.IONIC,
         14: rdkit.Chem.rdchem.BondType.HYDROGEN,
         15: rdkit.Chem.rdchem.BondType.THREECENTER,
         16: rdkit.Chem.rdchem.BondType.DATIVEONE,
         17: rdkit.Chem.rdchem.BondType.DATIVE,
         18: rdkit.Chem.rdchem.BondType.DATIVEL,
         19: rdkit.Chem.rdchem.BondType.DATIVER,
         20: rdkit.Chem.rdchem.BondType.OTHER,
         21: rdkit.Chem.rdchem.BondType.ZERO}
    """
    if kekulize:
        is_aromatic = False
        if bond.GetBondType().real == 12:
            bond_type = 1
        else:
            bond_type = bond.GetBondType().real
    else:
        is_aromatic = bond.GetIsAromatic()
        bond_type = bond.GetBondType().real
    return {'symbol': BondSymbol(is_aromatic=is_aromatic,
                                 bond_type=bond_type,
                                 stereo=int(bond.GetStereo())),
            'is_in_ring': bond.IsInRing(),
            'visited': False}


def standardize_stereo(mol):
    '''
 0: rdkit.Chem.rdchem.BondDir.NONE,
 1: rdkit.Chem.rdchem.BondDir.BEGINWEDGE,
 2: rdkit.Chem.rdchem.BondDir.BEGINDASH,
 3: rdkit.Chem.rdchem.BondDir.ENDDOWNRIGHT,
 4: rdkit.Chem.rdchem.BondDir.ENDUPRIGHT,

    '''
    # mol = Chem.AddHs(mol) # this removes CIPRank !!!
    for each_bond in mol.GetBonds():
        if int(each_bond.GetStereo()) in [2, 3]: #2=Z (same side), 3=E
            begin_stereo_atom_idx = each_bond.GetBeginAtomIdx()
            end_stereo_atom_idx = each_bond.GetEndAtomIdx()
            atom_idx_1 = each_bond.GetStereoAtoms()[0]
            atom_idx_2 = each_bond.GetStereoAtoms()[1]
            if mol.GetBondBetweenAtoms(atom_idx_1, begin_stereo_atom_idx):
                begin_atom_idx = atom_idx_1
                end_atom_idx = atom_idx_2
            else:
                begin_atom_idx = atom_idx_2
                end_atom_idx = atom_idx_1

            begin_another_atom_idx = None
            assert len(mol.GetAtomWithIdx(begin_stereo_atom_idx).GetNeighbors()) <= 3
            for each_neighbor in mol.GetAtomWithIdx(begin_stereo_atom_idx).GetNeighbors():
                each_neighbor_idx = each_neighbor.GetIdx()
                if each_neighbor_idx not in [end_stereo_atom_idx, begin_atom_idx]:
                    begin_another_atom_idx = each_neighbor_idx

            end_another_atom_idx = None
            assert len(mol.GetAtomWithIdx(end_stereo_atom_idx).GetNeighbors()) <= 3
            for each_neighbor in mol.GetAtomWithIdx(end_stereo_atom_idx).GetNeighbors():
                each_neighbor_idx = each_neighbor.GetIdx()
                if each_neighbor_idx not in [begin_stereo_atom_idx, end_atom_idx]:
                    end_another_atom_idx = each_neighbor_idx

            ''' 
            relationship between begin_atom_idx and end_atom_idx is encoded in GetStereo
            '''
            begin_atom_rank = int(mol.GetAtomWithIdx(begin_atom_idx).GetProp('_CIPRank'))
            end_atom_rank = int(mol.GetAtomWithIdx(end_atom_idx).GetProp('_CIPRank'))
            try:
                begin_another_atom_rank = int(mol.GetAtomWithIdx(begin_another_atom_idx).GetProp('_CIPRank'))
            except:
                begin_another_atom_rank = np.inf
            try:
                end_another_atom_rank = int(mol.GetAtomWithIdx(end_another_atom_idx).GetProp('_CIPRank'))
            except:
                end_another_atom_rank = np.inf
            if begin_atom_rank < begin_another_atom_rank\
               and end_atom_rank < end_another_atom_rank:
                pass
            elif begin_atom_rank < begin_another_atom_rank\
                 and end_atom_rank > end_another_atom_rank:
                # (begin_atom_idx +) end_another_atom_idx should be in StereoAtoms
                if each_bond.GetStereo() == 2:
                    # set stereo
                    each_bond.SetStereo(Chem.rdchem.BondStereo.values[3])
                    # set bond dir
                    mol = safe_set_bond_dir(mol, begin_atom_idx, begin_stereo_atom_idx, 3)
                    mol = safe_set_bond_dir(mol, begin_another_atom_idx, begin_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, end_atom_idx, end_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, end_another_atom_idx, end_stereo_atom_idx, 3)
                elif each_bond.GetStereo() == 3:
                    # set stereo
                    each_bond.SetStereo(Chem.rdchem.BondStereo.values[2])
                    # set bond dir
                    mol = safe_set_bond_dir(mol, begin_atom_idx, begin_stereo_atom_idx, 3)
                    mol = safe_set_bond_dir(mol, begin_another_atom_idx, begin_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, end_atom_idx, end_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, end_another_atom_idx, end_stereo_atom_idx, 4)
                else:
                    raise ValueError
                each_bond.SetStereoAtoms(begin_atom_idx, end_another_atom_idx)
            elif begin_atom_rank > begin_another_atom_rank\
                 and end_atom_rank < end_another_atom_rank:
                # (end_atom_idx +) begin_another_atom_idx should be in StereoAtoms
                if each_bond.GetStereo() == 2:
                    # set stereo
                    each_bond.SetStereo(Chem.rdchem.BondStereo.values[3])
                    # set bond dir
                    mol = safe_set_bond_dir(mol, begin_atom_idx, begin_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, begin_another_atom_idx, begin_stereo_atom_idx, 4)
                    mol = safe_set_bond_dir(mol, end_atom_idx, end_stereo_atom_idx, 4)
                    mol = safe_set_bond_dir(mol, end_another_atom_idx, end_stereo_atom_idx, 0)
                elif each_bond.GetStereo() == 3:
                    # set stereo
                    each_bond.SetStereo(Chem.rdchem.BondStereo.values[2])
                    # set bond dir
                    mol = safe_set_bond_dir(mol, begin_atom_idx, begin_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, begin_another_atom_idx, begin_stereo_atom_idx, 4)
                    mol = safe_set_bond_dir(mol, end_atom_idx, end_stereo_atom_idx, 3)
                    mol = safe_set_bond_dir(mol, end_another_atom_idx, end_stereo_atom_idx, 0)
                else:
                    raise ValueError
                each_bond.SetStereoAtoms(begin_another_atom_idx, end_atom_idx)
            elif begin_atom_rank > begin_another_atom_rank\
                 and end_atom_rank > end_another_atom_rank:
                # begin_another_atom_idx + end_another_atom_idx should be in StereoAtoms
                if each_bond.GetStereo() == 2:
                    # set bond dir
                    mol = safe_set_bond_dir(mol, begin_atom_idx, begin_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, begin_another_atom_idx, begin_stereo_atom_idx, 4)
                    mol = safe_set_bond_dir(mol, end_atom_idx, end_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, end_another_atom_idx, end_stereo_atom_idx, 3)
                elif each_bond.GetStereo() == 3:
                    # set bond dir
                    mol = safe_set_bond_dir(mol, begin_atom_idx, begin_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, begin_another_atom_idx, begin_stereo_atom_idx, 4)
                    mol = safe_set_bond_dir(mol, end_atom_idx, end_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, end_another_atom_idx, end_stereo_atom_idx, 4)
                else:
                    raise ValueError
                each_bond.SetStereoAtoms(begin_another_atom_idx, end_another_atom_idx)
            else:
                raise RuntimeError
    return mol


def set_stereo(mol):
    '''
 0: rdkit.Chem.rdchem.BondDir.NONE,
 1: rdkit.Chem.rdchem.BondDir.BEGINWEDGE,
 2: rdkit.Chem.rdchem.BondDir.BEGINDASH,
 3: rdkit.Chem.rdchem.BondDir.ENDDOWNRIGHT,
 4: rdkit.Chem.rdchem.BondDir.ENDUPRIGHT,
    '''
    _mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    Chem.Kekulize(_mol, True)
    substruct_match = mol.GetSubstructMatch(_mol)
    if not substruct_match:
        ''' mol and _mol are kekulized.
        sometimes, the order of '=' and '-' changes, which causes mol and _mol not matched.
        '''
        Chem.SetAromaticity(mol)
        Chem.SetAromaticity(_mol)
        substruct_match = mol.GetSubstructMatch(_mol)
    try:
        atom_match = {substruct_match[_mol_atom_idx]: _mol_atom_idx for _mol_atom_idx in range(_mol.GetNumAtoms())} # mol to _mol
    except:
        raise ValueError('two molecules obtained from the same data do not match.')
        
    for each_bond in mol.GetBonds():
        begin_atom_idx = each_bond.GetBeginAtomIdx()
        end_atom_idx = each_bond.GetEndAtomIdx()
        _bond = _mol.GetBondBetweenAtoms(atom_match[begin_atom_idx], atom_match[end_atom_idx])
        _bond.SetStereo(each_bond.GetStereo())

    mol = _mol
    for each_bond in mol.GetBonds():
        if int(each_bond.GetStereo()) in [2, 3]: #2=Z (same side), 3=E
            begin_stereo_atom_idx = each_bond.GetBeginAtomIdx()
            end_stereo_atom_idx = each_bond.GetEndAtomIdx()
            begin_atom_idx_set = set([each_neighbor.GetIdx()
                                      for each_neighbor
                                      in mol.GetAtomWithIdx(begin_stereo_atom_idx).GetNeighbors()
                                      if each_neighbor.GetIdx() != end_stereo_atom_idx])
            end_atom_idx_set = set([each_neighbor.GetIdx()
                                    for each_neighbor
                                    in mol.GetAtomWithIdx(end_stereo_atom_idx).GetNeighbors()
                                    if each_neighbor.GetIdx() != begin_stereo_atom_idx])
            if not begin_atom_idx_set:
                each_bond.SetStereo(Chem.rdchem.BondStereo(0))
                continue
            if not end_atom_idx_set:
                each_bond.SetStereo(Chem.rdchem.BondStereo(0))
                continue
            if len(begin_atom_idx_set) == 1:
                begin_atom_idx = begin_atom_idx_set.pop()
                begin_another_atom_idx = None
            if len(end_atom_idx_set) == 1:
                end_atom_idx = end_atom_idx_set.pop()
                end_another_atom_idx = None
            if len(begin_atom_idx_set) == 2:
                atom_idx_1 = begin_atom_idx_set.pop()
                atom_idx_2 = begin_atom_idx_set.pop()
                if int(mol.GetAtomWithIdx(atom_idx_1).GetProp('_CIPRank')) < int(mol.GetAtomWithIdx(atom_idx_2).GetProp('_CIPRank')):
                    begin_atom_idx = atom_idx_1
                    begin_another_atom_idx = atom_idx_2
                else:
                    begin_atom_idx = atom_idx_2
                    begin_another_atom_idx = atom_idx_1
            if len(end_atom_idx_set) == 2:
                atom_idx_1 = end_atom_idx_set.pop()
                atom_idx_2 = end_atom_idx_set.pop()
                if int(mol.GetAtomWithIdx(atom_idx_1).GetProp('_CIPRank')) < int(mol.GetAtomWithIdx(atom_idx_2).GetProp('_CIPRank')):
                    end_atom_idx = atom_idx_1
                    end_another_atom_idx = atom_idx_2
                else:
                    end_atom_idx = atom_idx_2
                    end_another_atom_idx = atom_idx_1

            if each_bond.GetStereo() == 2: # same side
                mol = safe_set_bond_dir(mol, begin_atom_idx, begin_stereo_atom_idx, 3)
                mol = safe_set_bond_dir(mol, end_atom_idx, end_stereo_atom_idx, 4)
                each_bond.SetStereoAtoms(begin_atom_idx, end_atom_idx)
            elif each_bond.GetStereo() == 3: # opposite side
                mol = safe_set_bond_dir(mol, begin_atom_idx, begin_stereo_atom_idx, 3)
                mol = safe_set_bond_dir(mol, end_atom_idx, end_stereo_atom_idx, 3)
                each_bond.SetStereoAtoms(begin_atom_idx, end_atom_idx)
            else:
                raise ValueError
    return mol


def safe_set_bond_dir(mol, atom_idx_1, atom_idx_2, bond_dir_val):
    if atom_idx_1 is None or atom_idx_2 is None:
        return mol
    else:
        mol.GetBondBetweenAtoms(atom_idx_1, atom_idx_2).SetBondDir(Chem.rdchem.BondDir.values[bond_dir_val])
        return mol
        
def common_node_list(hg1: Hypergraph, hg2: Hypergraph) -> List[str]:
    """ return a list of common nodes

    Parameters
    ----------
    hg1, hg2 : Hypergraph

    Returns
    -------
    list of str
        list of common nodes
    """
    if hg1 is None or hg2 is None:
        return [], False
    else:
        node_set = hg1.nodes.intersection(hg2.nodes)
        node_dict = {}
        if 'order4hrg' in hg1.node_attr(list(hg1.nodes)[0]):
            for each_node in node_set:
                node_dict[each_node] = hg1.node_attr(each_node)['order4hrg']
        else:
            for each_node in node_set:
                node_dict[each_node] = hg1.node_attr(each_node)['symbol'].__hash__()
        node_list = []
        for each_key, _ in sorted(node_dict.items(), key=lambda x:x[1]):
            node_list.append(each_key)
        edge_name = hg1.has_edge(node_list, ignore_order=True)
        if edge_name:
            if not hg1.edge_attr(edge_name).get('terminal', True):
                node_list = hg1.nodes_in_edge(edge_name)
            return node_list, True
        else:
            return node_list, False


