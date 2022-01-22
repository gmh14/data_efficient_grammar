from private.symbol import TSymbol, NTSymbol, BondSymbol
from private.utils import _node_match, _node_match_prod_rule, _edge_match, masked_softmax
from private.hypergraph import Hypergraph, common_node_list
from collections import Counter
from copy import deepcopy
from functools import partial
from abc import ABCMeta, abstractmethod
from networkx.algorithms.isomorphism import GraphMatcher
from typing import List, Dict, Tuple
import networkx as nx
import numpy as np
import torch
import os
import random
import warnings

DEBUG = False


class GraphGrammarBase(metaclass=ABCMeta):
    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def sample(self):
        pass


class ProductionRule(object):
    """ A class of a production rule

    Attributes
    ----------
    lhs : Hypergraph or None
        the left hand side of the production rule.
        if None, the rule is a starting rule.
    rhs : Hypergraph
        the right hand side of the production rule.
    """
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    @property
    def is_start_rule(self) -> bool:
        return self.lhs.num_nodes == 0

    @property
    def is_gluing(self) -> bool:
        if self.lhs:
            return len(self.lhs.get_all_NT_edges()) > 1
        return False

    @property
    def is_ending(self) -> bool:
        return len(self.rhs.get_all_NT_edges()) == 0

    @property
    def ext_node(self) -> Dict[int, str]:
        """ return a dict of external nodes
        """
        if self.is_start_rule:
            return {}
        else:
            ext_node_dict = {}
            for each_node in self.lhs.nodes:
                ext_node_dict[self.lhs.node_attr(each_node)["ext_id"]] = each_node
            return ext_node_dict

    @property
    def lhs_nt_symbol(self) -> NTSymbol:
        if self.is_start_rule:
            return NTSymbol(degree=0, is_aromatic=False, bond_symbol_list=[])
        else:
            # return self.lhs.edge_attr(list(self.lhs.edges)[0])['symbol']
            return [self.lhs.edge_attr(edge)['symbol'] for edge in list(self.lhs.edges)]

    def rhs_adj_mat(self, node_edge_list):
        ''' return the adjacency matrix of rhs of the production rule
        '''
        return nx.adjacency_matrix(self.rhs.hg, node_edge_list)

    def draw(self, file_path=None):
        return self.rhs.draw(file_path)

    def is_same(self, prod_rule, ignore_order=False):
        """ judge whether this production rule is
        the same as the input one, `prod_rule`

        Parameters
        ----------
        prod_rule : ProductionRule
            production rule to be compared

        Returns
        -------
        is_same : bool
        isomap : dict
            isomorphism of nodes and hyperedges.
            ex) {'bond_42': 'bond_37', 'bond_2': 'bond_1',
                 'e36': 'e11', 'e16': 'e12', 'e25': 'e18',
                 'bond_40': 'bond_38', 'e26': 'e21', 'bond_41': 'bond_39'}.
            key comes from `prod_rule`, value comes from `self`.
        """
        if self.is_start_rule:
            if not prod_rule.is_start_rule:
                return False, {}
        else:
            if prod_rule.is_start_rule:
                return False, {}
            else:
                if prod_rule.lhs.num_nodes != self.lhs.num_nodes:
                    return False, {}

        if prod_rule.rhs.num_nodes != self.rhs.num_nodes:
            return False, {}
        if prod_rule.rhs.num_edges != self.rhs.num_edges:
            return False, {}

        subhg_bond_symbol_counter \
                = Counter([prod_rule.rhs.node_attr(each_node)['symbol'] \
                for each_node in prod_rule.rhs.nodes])
        each_bond_symbol_counter \
                = Counter([self.rhs.node_attr(each_node)['symbol'] \
                for each_node in self.rhs.nodes])
        if subhg_bond_symbol_counter != each_bond_symbol_counter:
            return False, {}

        subhg_atom_symbol_counter \
                = Counter([prod_rule.rhs.edge_attr(each_edge)['symbol'] \
                for each_edge in prod_rule.rhs.edges])
        each_atom_symbol_counter \
                = Counter([self.rhs.edge_attr(each_edge)['symbol'] \
                for each_edge in self.rhs.edges])
        if subhg_atom_symbol_counter != each_atom_symbol_counter:
            return False, {}

        gm = GraphMatcher(prod_rule.rhs.hg,
                self.rhs.hg,
                partial(_node_match_prod_rule,
                    ignore_order=ignore_order),
                partial(_edge_match,
                    ignore_order=ignore_order))
        try:
            return True, next(gm.isomorphisms_iter())
        except StopIteration:
            return False, {}

    def graph_rule_applied_to(self, input_hg, selected_edge=None, selected_iso_mapping=None, vis=False):
        """ augment `hg` by replacing `edge` with `self.rhs`.

        Parameters
        ----------
        hg : Hypergraph
        edge : str
            `edge` must belong to `hg`

        Returns
        -------
        hg : Hypergraph
            resultant hypergraph
        nt_edge_list : list
            list of non-terminal edges
        """
        hg = deepcopy(input_hg)
        nt_edge_dict = {}
        if self.is_start_rule:
            node_idx = hg.num_nodes
            # hg = Hypergraph()
            node_map_rhs = {} # node id in rhs -> node id in hg, where rhs is augmented.
            for num_idx, each_node in enumerate(self.rhs.nodes):
                hg.add_node(f"bond_{num_idx+node_idx}",
                            attr_dict=self.rhs.node_attr(each_node))
                node_map_rhs[each_node] = f"bond_{num_idx+node_idx}"
            for each_edge in self.rhs.edges:
                node_list = []
                for each_node in self.rhs.nodes_in_edge(each_edge):
                    node_list.append(node_map_rhs[each_node])
                if isinstance(self.rhs.nodes_in_edge(each_edge), set):
                    node_list = set(node_list)
                edge_id = hg.add_edge(
                    node_list,
                    attr_dict=self.rhs.edge_attr(each_edge))
                if "nt_idx" in hg.edge_attr(edge_id):
                    nt_edge_dict[hg.edge_attr(edge_id)["nt_idx"]] = edge_id
            nt_edge_list = [nt_edge_dict[key] for key in range(len(nt_edge_dict))]
            return hg, nt_edge_list, True
        else:
            hg_NT_edges = hg.get_all_NT_edges()
            lhs_NT_edges = self.lhs.get_all_NT_edges()
            match_flag = []
            lhs_hg_matched_edge_dict = {}
            assert len(lhs_NT_edges) == 1

            # print("NTs:")
            # print([edge.edges for edge in hg_NT_edges])
            # print([edge.edge_attr(list(edge.edges)[0])['symbol'].symbol for edge in hg_NT_edges])
            # eq = lhs_NT_edges[0] == hg_NT_edges[0]

            for lhs_edge in lhs_NT_edges:
                if lhs_edge in hg_NT_edges:
                    match_flag.append(True)
                    assert len(lhs_edge.edges) == 1
                    for hg_NT_edge in hg_NT_edges:
                        if hg_NT_edge == lhs_edge:
                            if list(lhs_edge.edges)[0] not in lhs_hg_matched_edge_dict.keys():
                                lhs_hg_matched_edge_dict[list(lhs_edge.edges)[0]] = [hg_NT_edge]
                            else:
                                lhs_hg_matched_edge_dict[list(lhs_edge.edges)[0]].append(hg_NT_edge)
                else:
                    match_flag.append(False)

            if not all(match_flag):
                return hg, [], False

            # for edge in hg.edges:
                # print("edge: {} -> nodes: {}".format(edge, hg.nodes_in_edge(edge)))

            # order of nodes that belong to the non-terminal edge in hg
            nt_order_dict = {}  # hg_node -> order ("bond_17" : 1)
            nt_order_dict_inv = {} # order -> hg_node
            to_rm_edges = []
            assert len(lhs_NT_edges) == 1
            for _i, edge_lhs in enumerate(lhs_NT_edges):
                edges_hg = lhs_hg_matched_edge_dict[list(edge_lhs.edges)[0]]
                edges_name_to_hg = {list(edge_hg.edges)[0]:edge_hg for edge_hg in edges_hg}
                edges_cand = edges_name_to_hg.keys()
                # filter out those removed edges in previous iterations
                edges = [_edge for _edge in edges_cand if _edge not in to_rm_edges]
                # if it is empty, meaning that there is only actually one NT in hg that matches multiple NTs in the lhs, the rule should be abandoned
                if len(edges) == 0:
                    return hg, [], False
                # TODO add options
                edge = selected_edge
                if edge == None:
                    edge = np.random.choice(edges, 1)[0]
                # print('selected edge:', edge)
                # edge_hg = edges_hg[edges_cand.index(edge)]
                edge_hg = edges_name_to_hg[edge]
                iso_mappings = edge_hg.find_isomorphism_mapping(edge_lhs, vis) # From edge_hg to edge_lhs
                # print("mapping:", iso_mapping)
                if selected_iso_mapping is None:
                    iso_mapping = np.random.choice(iso_mappings, 1)[0]
                else:
                    iso_mapping = selected_iso_mapping

                for each_idx, each_node in enumerate(hg.nodes_in_edge(edge)):
                    mapped_node_in_lhs = iso_mapping[each_node]
                    ext_id = edge_lhs.node_attr(mapped_node_in_lhs)['ext_id']
                    nt_order_dict[each_node] = ext_id
                    nt_order_dict_inv[ext_id] = each_node
                to_rm_edges.append(edge)

            for edge in to_rm_edges:
                # delete non-terminal
                hg.remove_edge(edge)

            # construct a node_map_rhs: rhs -> new hg
            node_map_rhs = {} # node id in rhs -> node id in hg, where rhs is augmented.
            node_idx = hg.num_nodes
            for each_node in self.rhs.nodes:
                if "ext_id" in self.rhs.node_attr(each_node):
                    node_map_rhs[each_node] \
                            = nt_order_dict_inv[
                                    self.rhs.node_attr(each_node)["ext_id"]]
                else:
                    node_map_rhs[each_node] = f"bond_{node_idx}"
                    node_idx += 1

            # add nodes to hg
            for each_node in self.rhs.nodes:
                hg.add_node(node_map_rhs[each_node],
                        attr_dict=self.rhs.node_attr(each_node))

            # add hyperedges to hg
            for each_edge in self.rhs.edges:
                node_list_hg = []
                for each_node in self.rhs.nodes_in_edge(each_edge):
                    node_list_hg.append(node_map_rhs[each_node])
                edge_id = hg.add_edge(
                        node_list_hg,
                        attr_dict=self.rhs.edge_attr(each_edge))#deepcopy(self.rhs.edge_attr(each_edge)))
                if "nt_idx" in hg.edge_attr(edge_id):
                    nt_edge_dict[hg.edge_attr(edge_id)["nt_idx"]] = edge_id
            nt_edge_list = [nt_edge_dict[key] for key in range(len(nt_edge_dict))]


            return hg, nt_edge_list, True

    def get_all_compatible_edges(self, input_hg):
        hg = deepcopy(input_hg)
        nt_edge_dict = {}
        assert not self.is_start_rule
        hg_NT_edges = hg.get_all_NT_edges()
        lhs_NT_edges = self.lhs.get_all_NT_edges()
        match_flag = []
        lhs_hg_matched_edge_dict = {}

        for lhs_edge in lhs_NT_edges:
            if lhs_edge in hg_NT_edges:
                match_flag.append(True)
                assert len(lhs_edge.edges) == 1
                for hg_NT_edge in hg_NT_edges:
                    if hg_NT_edge == lhs_edge:
                        if list(lhs_edge.edges)[0] not in lhs_hg_matched_edge_dict.keys():
                            lhs_hg_matched_edge_dict[list(lhs_edge.edges)[0]] = [hg_NT_edge]
                        else:
                            lhs_hg_matched_edge_dict[list(lhs_edge.edges)[0]].append(hg_NT_edge)
            else:
                match_flag.append(False)

        if not all(match_flag):
            return hg, [], False

        # order of nodes that belong to the non-terminal edge in hg
        nt_order_dict = {}  # hg_node -> order ("bond_17" : 1)
        nt_order_dict_inv = {} # order -> hg_node
        to_rm_edges = []
        selected_edges = []
        selected_iso_mappings = []
        for _i, edge_lhs in enumerate(lhs_NT_edges):
            edges_hg = lhs_hg_matched_edge_dict[list(edge_lhs.edges)[0]]
            edges_name_to_hg = {list(edge_hg.edges)[0]:edge_hg for edge_hg in edges_hg}
            edges_cand = edges_name_to_hg.keys()
            # filter out those removed edges in previous iterations
            edges = [_edge for _edge in edges_cand if _edge not in to_rm_edges]
            # if it is empty, meaning that there is only actually one NT in hg that matches multiple NTs in the lhs, the rule should be abandoned
            if len(edges) == 0:
                return hg, [], False
            # edge = np.random.choice(edges, 1)[0]
            for edge in edges:
                edge_hg = edges_name_to_hg[edge]
                iso_mappings = edge_hg.find_isomorphism_mapping(edge_lhs) # From edge_hg to edge_lhs
                # print("mapping:", iso_mapping)
                selected_iso_mappings.append(iso_mappings)
            selected_edges.extend(edges)
        return selected_edges, selected_iso_mappings
                

    def applied_to(self,
            hg: Hypergraph,
            edge: str) -> Tuple[Hypergraph, List[str]]:
        """ augment `hg` by replacing `edge` with `self.rhs`.

        Parameters
        ----------
        hg : Hypergraph
        edge : str
            `edge` must belong to `hg`

        Returns
        -------
        hg : Hypergraph
            resultant hypergraph
        nt_edge_list : list
            list of non-terminal edges
        """
        nt_edge_dict = {}
        if self.is_start_rule:
            if (edge is not None) or (hg is not None):
                ValueError("edge and hg must be None for this prod rule.")
            hg = Hypergraph()
            node_map_rhs = {} # node id in rhs -> node id in hg, where rhs is augmented.
            for num_idx, each_node in enumerate(self.rhs.nodes):
                hg.add_node(f"bond_{num_idx}",
                        #attr_dict=deepcopy(self.rhs.node_attr(each_node)))
                        attr_dict=self.rhs.node_attr(each_node))
                node_map_rhs[each_node] = f"bond_{num_idx}"
            for each_edge in self.rhs.edges:
                node_list = []
                for each_node in self.rhs.nodes_in_edge(each_edge):
                    node_list.append(node_map_rhs[each_node])
                if isinstance(self.rhs.nodes_in_edge(each_edge), set):
                    node_list = set(node_list)
                edge_id = hg.add_edge(
                        node_list,
                        #attr_dict=deepcopy(self.rhs.edge_attr(each_edge)))
                        attr_dict=self.rhs.edge_attr(each_edge))
                if "nt_idx" in hg.edge_attr(edge_id):
                    nt_edge_dict[hg.edge_attr(edge_id)["nt_idx"]] = edge_id
            nt_edge_list = [nt_edge_dict[key] for key in range(len(nt_edge_dict))]
            return hg, nt_edge_list
        else:
            if edge not in hg.edges:
                raise ValueError("the input hyperedge does not exist.")
            if hg.edge_attr(edge)["terminal"]:
                raise ValueError("the input hyperedge is terminal.")
            if hg.edge_attr(edge)['symbol'] != self.lhs_nt_symbol:
                print(hg.edge_attr(edge)['symbol'], self.lhs_nt_symbol)
                raise ValueError("the input hyperedge and lhs have inconsistent number of nodes.")
            if DEBUG:
                for node_idx, each_node in enumerate(hg.nodes_in_edge(edge)):
                    other_node = self.lhs.nodes_in_edge(list(self.lhs.edges)[0])[node_idx]
                    attr = deepcopy(self.lhs.node_attr(other_node))
                    attr.pop('ext_id')
                    if hg.node_attr(each_node) != attr:
                        raise ValueError('node attributes are inconsistent.')

            # order of nodes that belong to the non-terminal edge in hg
            nt_order_dict = {}  # hg_node -> order ("bond_17" : 1)
            nt_order_dict_inv = {} # order -> hg_node
            for each_idx, each_node in enumerate(hg.nodes_in_edge(edge)):
                nt_order_dict[each_node] = each_idx
                nt_order_dict_inv[each_idx] = each_node

            # construct a node_map_rhs: rhs -> new hg
            node_map_rhs = {} # node id in rhs -> node id in hg, where rhs is augmented.
            node_idx = hg.num_nodes
            for each_node in self.rhs.nodes:
                if "ext_id" in self.rhs.node_attr(each_node):
                    node_map_rhs[each_node] \
                            = nt_order_dict_inv[
                                    self.rhs.node_attr(each_node)["ext_id"]]
                else:
                    node_map_rhs[each_node] = f"bond_{node_idx}"
                    node_idx += 1

            # delete non-terminal
            hg.remove_edge(edge)

            # add nodes to hg
            for each_node in self.rhs.nodes:
                hg.add_node(node_map_rhs[each_node],
                        attr_dict=self.rhs.node_attr(each_node))

            # add hyperedges to hg
            for each_edge in self.rhs.edges:
                node_list_hg = []
                for each_node in self.rhs.nodes_in_edge(each_edge):
                    node_list_hg.append(node_map_rhs[each_node])
                edge_id = hg.add_edge(
                        node_list_hg,
                        attr_dict=self.rhs.edge_attr(each_edge))#deepcopy(self.rhs.edge_attr(each_edge)))
                if "nt_idx" in hg.edge_attr(edge_id):
                    nt_edge_dict[hg.edge_attr(edge_id)["nt_idx"]] = edge_id
            nt_edge_list = [nt_edge_dict[key] for key in range(len(nt_edge_dict))]
            return hg, nt_edge_list

    def revert(self, hg: Hypergraph, return_subhg=False):
        ''' revert applying this production rule.
        i.e., if there exists a subhypergraph that matches the r.h.s. of this production rule,
        this method replaces the subhypergraph with a non-terminal hyperedge.

        Parameters
        ----------
        hg : Hypergraph
            hypergraph to be reverted
        return_subhg : bool
            if True, the removed subhypergraph will be returned.

        Returns
        -------
        hg : Hypergraph
            the resultant hypergraph. if it cannot be reverted, the original one is returned without any replacement.
        success : bool
            this indicates whether reverting is successed or not.
        '''
        gm = GraphMatcher(hg.hg, self.rhs.hg, node_match=_node_match_prod_rule,
                edge_match=_edge_match)
        try:
            # in case when the matched subhg is connected to the other part via external nodes and more.
            not_iso = True
            while not_iso:
                isomap = next(gm.subgraph_isomorphisms_iter())
                adj_node_set = set([]) # reachable nodes from the internal nodes
                subhg_node_set = set(isomap.keys()) # nodes in subhg
                for each_node in subhg_node_set:
                    adj_node_set.add(each_node)
                    if isomap[each_node] not in self.ext_node.values():
                        adj_node_set.update(hg.hg.adj[each_node])
                if adj_node_set == subhg_node_set:
                    not_iso = False
                else:
                    if return_subhg:
                        return hg, False, Hypergraph()
                    else:
                        return hg, False
            inv_isomap = {v: k for k, v in isomap.items()}
            '''
            isomap = {'e35': 'e8', 'bond_13': 'bond_18', 'bond_14': 'bond_19',
                      'bond_15': 'bond_17', 'e29': 'e23', 'bond_12': 'bond_20'}
            where keys come from `hg` and values come from `self.rhs`
            '''
        except StopIteration:
            if return_subhg:
                return hg, False, Hypergraph()
            else:
                return hg, False

        if return_subhg:
            subhg = Hypergraph()
            for each_node in hg.nodes:
                if each_node in isomap:
                    subhg.add_node(each_node, attr_dict=hg.node_attr(each_node))
            for each_edge in hg.edges:
                if each_edge in isomap:
                    subhg.add_edge(hg.nodes_in_edge(each_edge),
                            attr_dict=hg.edge_attr(each_edge),
                            edge_name=each_edge)
            subhg.edge_idx = hg.edge_idx

        # remove subhg except for the externael nodes
        for each_key, each_val in isomap.items():
            if each_key.startswith('e'):
                hg.remove_edge(each_key)
        for each_key, each_val in isomap.items():
            if each_key.startswith('bond_'):
                if each_val not in self.ext_node.values():
                    hg.remove_node(each_key)

        # add non-terminal hyperedge
        nt_node_list = []
        for each_ext_id in self.ext_node.keys():
            nt_node_list.append(inv_isomap[self.ext_node[each_ext_id]])

        hg.add_edge(nt_node_list,
                attr_dict=dict(
                    terminal=False,
                    symbol=self.lhs_nt_symbol))
        if return_subhg:
            return hg, True, subhg
        else:
            return hg, True


class ProductionRuleCorpus(object):

    '''
    A corpus of production rules.
    This class maintains 
        (i) list of unique production rules,
        (ii) list of unique edge symbols (both terminal and non-terminal), and
        (iii) list of unique node symbols.

    Attributes
    ----------
    prod_rule_list : list
        list of unique production rules
    edge_symbol_list : list
        list of unique symbols (including both terminal and non-terminal)
    node_symbol_list : list
        list of node symbols
    nt_symbol_list : list
        _ist of unique lhs symbols
    ext_id_list : list
        list of ext_ids
    lhs_in_prod_rule : array
        a matrix of lhs vs prod_rule (= lhs_in_prod_rule)
    '''

    def __init__(self):
        self.prod_rule_list = []
        self.edge_symbol_list = []
        self.edge_symbol_dict = {}
        self.node_symbol_list = []
        self.node_symbol_dict = {}
        self.nt_symbol_list = []
        self.ext_id_list = []
        self._lhs_in_prod_rule = None
        self.lhs_in_prod_rule_row_list = []
        self.lhs_in_prod_rule_col_list = []
        self.lhs_list = []

    @property
    def lhs_in_prod_rule(self):
        if self._lhs_in_prod_rule is None:
            self._lhs_in_prod_rule = torch.sparse.FloatTensor(
                    torch.LongTensor(list(zip(self.lhs_in_prod_rule_row_list, self.lhs_in_prod_rule_col_list))).t(),
                    torch.FloatTensor([1.0]*len(self.lhs_in_prod_rule_col_list)),
                    torch.Size([len(self.nt_symbol_list), len(self.prod_rule_list)])
                    ).to_dense()
        return self._lhs_in_prod_rule

    @property
    def num_prod_rule(self):
        ''' return the number of production rules

        Returns
        -------
        int : the number of unique production rules
        '''
        return len(self.prod_rule_list)

    @property
    def start_rule_list(self):
        ''' return a list of start rules

        Returns
        -------
        list : list of start rules
        '''
        start_rule_list = []
        for each_prod_rule in self.prod_rule_list:
            if each_prod_rule.is_start_rule:
                start_rule_list.append(each_prod_rule)
        return start_rule_list

    @property
    def num_edge_symbol(self):
        return len(self.edge_symbol_list)

    @property
    def num_node_symbol(self):
        return len(self.node_symbol_list)

    @property
    def num_ext_id(self):
        return len(self.ext_id_list)

    def construct_feature_vectors(self):
        ''' this method constructs feature vectors for the production rules collected so far.
        currently, NTSymbol and TSymbol are treated in the same manner.
        '''
        feature_id_dict = {}
        feature_id_dict['TSymbol'] = 0
        feature_id_dict['NTSymbol'] = 1
        feature_id_dict['BondSymbol'] = 2
        for each_edge_symbol in self.edge_symbol_list:
            for each_attr in each_edge_symbol.__dict__.keys():
                each_val = each_edge_symbol.__dict__[each_attr]
                if isinstance(each_val, list):
                    each_val = tuple(each_val)
                if (each_attr, each_val) not in feature_id_dict:
                    feature_id_dict[(each_attr, each_val)] = len(feature_id_dict)

        for each_node_symbol in self.node_symbol_list:
            for each_attr in each_node_symbol.__dict__.keys():
                each_val = each_node_symbol.__dict__[each_attr]
                if isinstance(each_val, list):
                    each_val = tuple(each_val)
                if (each_attr, each_val) not in feature_id_dict:
                    feature_id_dict[(each_attr, each_val)] = len(feature_id_dict)
        for each_ext_id in self.ext_id_list:
            feature_id_dict[('ext_id', each_ext_id)] = len(feature_id_dict)
        dim = len(feature_id_dict)

        feature_dict = {}
        for each_edge_symbol in self.edge_symbol_list:
            idx_list = []
            idx_list.append(feature_id_dict[each_edge_symbol.__class__.__name__])
            for each_attr in each_edge_symbol.__dict__.keys():
                each_val = each_edge_symbol.__dict__[each_attr]
                if isinstance(each_val, list):
                    each_val = tuple(each_val)
                idx_list.append(feature_id_dict[(each_attr, each_val)])
            feature = torch.sparse.LongTensor(
                    torch.LongTensor([idx_list]),
                    torch.ones(len(idx_list)),
                    torch.Size([len(feature_id_dict)])
                    )
            feature_dict[each_edge_symbol] = feature

        for each_node_symbol in self.node_symbol_list:
            idx_list = []
            idx_list.append(feature_id_dict[each_node_symbol.__class__.__name__])
            for each_attr in each_node_symbol.__dict__.keys():
                each_val = each_node_symbol.__dict__[each_attr]
                if isinstance(each_val, list):
                    each_val = tuple(each_val)
                idx_list.append(feature_id_dict[(each_attr, each_val)])
            feature = torch.sparse.LongTensor(
                    torch.LongTensor([idx_list]),
                    torch.ones(len(idx_list)),
                    torch.Size([len(feature_id_dict)])
                    )
            feature_dict[each_node_symbol] = feature
        for each_ext_id in self.ext_id_list:
            idx_list = [feature_id_dict[('ext_id', each_ext_id)]]
            feature_dict[('ext_id', each_ext_id)] \
                    = torch.sparse.LongTensor(
                            torch.LongTensor([idx_list]),
                            torch.ones(len(idx_list)),
                            torch.Size([len(feature_id_dict)])
                            )
        return feature_dict, dim

    def edge_symbol_idx(self, symbol):
        return self.edge_symbol_dict[symbol]

    def node_symbol_idx(self, symbol):
        return self.node_symbol_dict[symbol]

    def append(self, prod_rule: ProductionRule) -> Tuple[int, ProductionRule]:
        """ return whether the input production rule is new or not, and its production rule id.
        Production rules are regarded as the same if 
            i) there exists a one-to-one mapping of nodes and edges, and
            ii) all the attributes associated with nodes and hyperedges are the same.

        Parameters
        ----------
        prod_rule : ProductionRule

        Returns
        -------
        prod_rule_id : int
            production rule index. if new, a new index will be assigned.
        prod_rule : ProductionRule
        """
        num_lhs = len(self.nt_symbol_list)
        for each_idx, each_prod_rule in enumerate(self.prod_rule_list):
            is_same, isomap = prod_rule.is_same(each_prod_rule, ignore_order=True)
            if is_same:
                # we do not care about edge and node names, but care about the order of non-terminal edges.
                for key, val in isomap.items(): # key : edges & nodes in each_prod_rule.rhs , val : those in prod_rule.rhs
                    if key.startswith("bond_"):
                        continue

                    # rewrite `nt_idx` in `prod_rule` for further processing
                    if "nt_idx" in prod_rule.rhs.edge_attr(val).keys():
                        if "nt_idx" not in each_prod_rule.rhs.edge_attr(key).keys():
                            raise ValueError
                        prod_rule.rhs.set_edge_attr(
                            val,
                            {'nt_idx': each_prod_rule.rhs.edge_attr(key)["nt_idx"]})
                return each_idx, prod_rule
        self.prod_rule_list.append(prod_rule)
        self.lhs_list.append(prod_rule.lhs)
        self._update_edge_symbol_list(prod_rule)
        self._update_node_symbol_list(prod_rule)
        self._update_ext_id_list(prod_rule)

        lhs_idx = self.nt_symbol_list.index(prod_rule.lhs_nt_symbol)
        self.lhs_in_prod_rule_row_list.append(lhs_idx)
        self.lhs_in_prod_rule_col_list.append(len(self.prod_rule_list)-1)
        self._lhs_in_prod_rule = None
        return len(self.prod_rule_list)-1, prod_rule

    def get_prod_rules_with_lhs(self, lhs):
        assert isinstance(lhs, Hypergraph)
        prod_rules = []
        for i, _lhs in enumerate(self.lhs_list):
            if lhs.is_same(_lhs, ignore_order=True):
                prod_rules.append(self.get_prod_rule(i))
        return prod_rules

    def get_prod_rule(self, prod_rule_idx: int) -> ProductionRule:
        return self.prod_rule_list[prod_rule_idx]

    def sample(self, unmasked_logit_array, nt_symbol, deterministic=False):
        ''' sample a production rule whose lhs is `nt_symbol`, followihng `unmasked_logit_array`.

        Parameters
        ----------
        unmasked_logit_array : array-like, length `num_prod_rule`
        nt_symbol : NTSymbol
        '''
        if not isinstance(unmasked_logit_array, np.ndarray):
            unmasked_logit_array = unmasked_logit_array.numpy().astype(np.float64)
        if deterministic:
            prob = masked_softmax(unmasked_logit_array,
                                  self.lhs_in_prod_rule[self.nt_symbol_list.index(nt_symbol)].numpy().astype(np.float64))
            return self.prod_rule_list[np.argmax(prob)]
        else:
            return np.random.choice(
                self.prod_rule_list, 1,
                p=masked_softmax(unmasked_logit_array,
                                 self.lhs_in_prod_rule[self.nt_symbol_list.index(nt_symbol)].numpy().astype(np.float64)))[0]

    def _update_edge_symbol_list(self, prod_rule: ProductionRule):
        ''' update edge symbol list

        Parameters
        ----------
        prod_rule : ProductionRule
        '''
        if prod_rule.lhs_nt_symbol not in self.nt_symbol_list:
            self.nt_symbol_list.append(prod_rule.lhs_nt_symbol)

        for each_edge in prod_rule.rhs.edges:
            if prod_rule.rhs.edge_attr(each_edge)['symbol'] not in self.edge_symbol_dict:
                edge_symbol_idx = len(self.edge_symbol_list)
                self.edge_symbol_list.append(prod_rule.rhs.edge_attr(each_edge)['symbol'])
                self.edge_symbol_dict[prod_rule.rhs.edge_attr(each_edge)['symbol']] = edge_symbol_idx
            else:
                edge_symbol_idx = self.edge_symbol_dict[prod_rule.rhs.edge_attr(each_edge)['symbol']]
            prod_rule.rhs.edge_attr(each_edge)['symbol_idx'] = edge_symbol_idx
        pass

    def _update_node_symbol_list(self, prod_rule: ProductionRule):
        ''' update node symbol list

        Parameters
        ----------
        prod_rule : ProductionRule
        '''
        for each_node in prod_rule.rhs.nodes:
            if prod_rule.rhs.node_attr(each_node)['symbol'] not in self.node_symbol_dict:
                node_symbol_idx = len(self.node_symbol_list)
                self.node_symbol_list.append(prod_rule.rhs.node_attr(each_node)['symbol'])
                self.node_symbol_dict[prod_rule.rhs.node_attr(each_node)['symbol']] = node_symbol_idx
            else:
                node_symbol_idx = self.node_symbol_dict[prod_rule.rhs.node_attr(each_node)['symbol']]
            prod_rule.rhs.node_attr(each_node)['symbol_idx'] = node_symbol_idx

    def _update_ext_id_list(self, prod_rule: ProductionRule):
        for each_node in prod_rule.rhs.nodes:
            if 'ext_id' in prod_rule.rhs.node_attr(each_node):
                if prod_rule.rhs.node_attr(each_node)['ext_id'] not in self.ext_id_list:
                    self.ext_id_list.append(prod_rule.rhs.node_attr(each_node)['ext_id'])


def reorder_children(myself, children, prod_rule, clique_tree):
    """ reorder children so that they match the order in `prod_rule`.

    Parameters
    ----------
    myself : int
    children : list of int
    prod_rule : ProductionRule
    clique_tree : nx.Graph

    Returns
    -------
    new_children : list of str
        reordered children
    """
    perm = {} # key : `nt_idx`, val : child
    for each_edge in prod_rule.rhs.edges:
        if "nt_idx" in prod_rule.rhs.edge_attr(each_edge).keys():
            for each_child in children:
                common_node_set = set(
                    common_node_list(clique_tree.node[myself]["subhg"],
                                     clique_tree.node[each_child]["subhg"])[0])
                if set(prod_rule.rhs.nodes_in_edge(each_edge)) == common_node_set:
                    assert prod_rule.rhs.edge_attr(each_edge)["nt_idx"] not in perm
                    perm[prod_rule.rhs.edge_attr(each_edge)["nt_idx"]] = each_child
    new_children = []
    assert len(perm) == len(children)
    for i in range(len(perm)):
        new_children.append(perm[i])
    return new_children


def generate_rule(input_graph, subg, grammar):
    def _add_ext_node(hg, ext_nodes):
        """ mark nodes to be external (ordered ids are assigned)

        Parameters
        ----------
        hg : UndirectedHypergraph
        ext_nodes : list of str
            list of external nodes

        Returns
        -------
        hg : Hypergraph
            nodes in `ext_nodes` are marked to be external
        """
        ext_id = 0
        ext_id_exists = []
        for each_node in ext_nodes:
            ext_id_exists.append('ext_id' in hg.node_attr(each_node))
        if ext_id_exists and any(ext_id_exists) != all(ext_id_exists):
            raise ValueError
        if not all(ext_id_exists):
            for each_node in ext_nodes:
                hg.node_attr(each_node)['ext_id'] = ext_id
                ext_id += 1
        return hg

    def _check_aromatic(hg, node_list):
        is_aromatic = False
        node_aromatic_list = []
        for each_node in node_list:
            if hg.node_attr(each_node)['symbol'].is_aromatic:
                is_aromatic = True
                node_aromatic_list.append(True)
            else:
                node_aromatic_list.append(False)
        return is_aromatic, node_aromatic_list

    def _check_ring(hg):
        for each_edge in hg.edges:
            if not ('tmp' in hg.edge_attr(each_edge) or (not hg.edge_attr(each_edge)['terminal'])):
                return False
        return True
    
    ''' high-level idea:
    find all ext_nodes: for each edge with [:1], find its adj nodes, for those nodes that are not included in the subg, set them as ext_node
    lhs: single NT with ext_nodes
    rhs: all visited edges shrink to NT
    '''
    input_graph_hg = input_graph.hypergraph
    subg_hyper_graph = subg.hypergraph
    ext_node_list = set()
    watershed_mapping = {}
    all_nodes_subg = [input_graph.get_org_node_in_input(node, subg) for node in subg_hyper_graph.nodes] # TODO a stupid implementation

    # rhs
    rhs = Hypergraph()
    # Several goals to achieve during the loop
    # for rhs use: make a copy of subg_hyper_graph, with all attr replaced with attr from original input_g, connect ext_nodes to the hypergraph
    # for lhs use: get the list of ext_nodes
    for edge in subg_hyper_graph.edges:
        # Map all the nodes and edges to the input to get latest status&attribute
        # Get the original name of edge
        org_edge = 'e{}'.format(subg.get_org_idx_in_input(int(edge[1:])))
        nodes = subg_hyper_graph.nodes_in_edge(edge)

        # Get all the original names of nodes for the current subg
        node_list_subg = []
        for node in nodes:
            org_node = input_graph.get_org_node_in_input(node, subg)
            node_list_subg.append(org_node)

        if not input_graph.get_visit_status_edge(org_edge):
            # for unvisited edge
            node_list = [] # Nodes to be added to the rhs
            for node in nodes:
                org_node = input_graph.get_org_node_in_input(node, subg)
                # only need to add unvisited nodes
                if not input_graph_hg.node_attr(org_node)['visited']: 
                    rhs.add_node(org_node, deepcopy(input_graph_hg.node_attr(org_node)))
                    node_list.append(org_node)
            if not subg_hyper_graph.edge_attr(edge)['terminal']:
                # for those unvisited edges with [:1], add ext nodes with original names if any, and update node_list for the use of rhs
                for _node in input_graph_hg.nodes_in_edge(org_edge):
                    # ext_node added
                    if not (_node in node_list_subg):
                        ext_node_list.add(_node)
                        rhs.add_node(_node, deepcopy(input_graph_hg.node_attr(_node)))
                        node_list.append(_node)
            edge_attr_dict = deepcopy(input_graph_hg.edge_attr(org_edge))
            rhs.add_edge(node_list, attr_dict=edge_attr_dict)
        else:
            # for visited edge
            # each visited edge will have a water_level attribute
            # record the unvisited nodes and ext_nodes
            edge_water_level = input_graph_hg.edge_attr(org_edge)['water_level']
            nodes_to_add = deepcopy(input_graph.watershed_ext_nodes[edge_water_level])
            if edge_water_level not in watershed_mapping.keys():
                watershed_mapping[edge_water_level] = set(nodes_to_add)
            else:
                watershed_mapping[edge_water_level].update(nodes_to_add)
            for _node in nodes_to_add:
                if not (_node in all_nodes_subg):
                    ext_node_list.add(_node)

    # Add necessary edges and nodes for the visited edges
    for water_level, node_set in watershed_mapping.items():
        node_list = list(node_set)
        is_aromatic, _ = _check_aromatic(input_graph_hg, node_list)
        bond_symbol_list = []
        for each_node in node_list:
            rhs.add_node(each_node, deepcopy(input_graph_hg.node_attr(each_node)))
            bond_symbol_list.append(input_graph_hg.node_attr(each_node)['symbol'])
        edge_attr_dict = dict(terminal=False,
                        symbol=NTSymbol(
                            degree=len(node_list),
                            is_aromatic=is_aromatic,
                            bond_symbol_list=bond_symbol_list))
        rhs.add_edge(node_list, attr_dict=edge_attr_dict)
    try:
        rhs = _add_ext_node(rhs, ext_node_list)
    except ValueError:
        import pdb; pdb.set_trace()

    # lhs
    starting = False
    # If no ext_nodes in un_visited_edges, then it should be a starting rule
    if len(ext_node_list) == 0:
        starting = True
    lhs = Hypergraph()
    if not starting:
        bond_symbol_list = []
        for each_node in ext_node_list:
            lhs.add_node(each_node, deepcopy(input_graph_hg.node_attr(each_node)))
            bond_symbol_list.append(input_graph_hg.node_attr(each_node)['symbol'])
        is_aromatic, _ = _check_aromatic(input_graph_hg, ext_node_list)
        lhs.add_edge(
            ext_node_list,
            attr_dict=dict(
                terminal=False,
                # edge_exists=edge_exists,
                symbol=NTSymbol(
                    degree=len(ext_node_list),
                    is_aromatic=is_aromatic,
                    bond_symbol_list=bond_symbol_list)))
        try:
            lhs = _add_ext_node(lhs, ext_node_list)
        except ValueError:
            import pdb; pdb.set_trace()
    
    if lhs != rhs:
        # looks crazy, but still possible
        # one possible cause: all the edges&atoms of the selected subg has been visited, because two previously selected subgs consist of the current subg. E.g. [*]C(C)C, the two previously selected subgs are C-C
        rule = ProductionRule(lhs, rhs)
        rule_idx, _ = grammar.append(rule)
        input_graph.append_rule(rule, rule_idx)

    return grammar

def extract_prod_rule(parent_hg, myself_hg, children_hg_list, subhg_idx=None):
    """ extract a production rule from a triple of `parent_hg`, `myself_hg`, and `children_hg_list`.

    Parameters
    ----------
    parent_hg : Hypergraph
    myself_hg : Hypergraph
    children_hg_list : list of Hypergraph

    Returns
    -------
    ProductionRule, consisting of
        lhs : Hypergraph or None
        rhs : Hypergraph
    """
    def _add_ext_node(hg, ext_nodes):
        """ mark nodes to be external (ordered ids are assigned)

        Parameters
        ----------
        hg : UndirectedHypergraph
        ext_nodes : list of str
            list of external nodes

        Returns
        -------
        hg : Hypergraph
            nodes in `ext_nodes` are marked to be external
        """
        ext_id = 0
        ext_id_exists = []
        for each_node in ext_nodes:
            ext_id_exists.append('ext_id' in hg.node_attr(each_node))
        if ext_id_exists and any(ext_id_exists) != all(ext_id_exists):
            raise ValueError
        if not all(ext_id_exists):
            for each_node in ext_nodes:
                hg.node_attr(each_node)['ext_id'] = ext_id
                ext_id += 1
        return hg

    def _check_aromatic(hg, node_list):
        is_aromatic = False
        node_aromatic_list = []
        for each_node in node_list:
            if hg.node_attr(each_node)['symbol'].is_aromatic:
                is_aromatic = True
                node_aromatic_list.append(True)
            else:
                node_aromatic_list.append(False)
        return is_aromatic, node_aromatic_list

    def _check_ring(hg):
        for each_edge in hg.edges:
            if not ('tmp' in hg.edge_attr(each_edge) or (not hg.edge_attr(each_edge)['terminal'])):
                return False
        return True

    if parent_hg is None:
        lhs = Hypergraph()
        node_list = []
    else:
        lhs = Hypergraph()
        node_list, edge_exists = common_node_list(parent_hg, myself_hg)
        for each_node in node_list:
            lhs.add_node(each_node,
                         deepcopy(myself_hg.node_attr(each_node)))
        is_aromatic, _ = _check_aromatic(parent_hg, node_list)
        for_ring = _check_ring(myself_hg)
        bond_symbol_list = []
        for each_node in node_list:
            bond_symbol_list.append(parent_hg.node_attr(each_node)['symbol'])
        lhs.add_edge(
            node_list,
            attr_dict=dict(
                terminal=False,
                edge_exists=edge_exists,
                symbol=NTSymbol(
                    degree=len(node_list),
                    is_aromatic=is_aromatic,
                    bond_symbol_list=bond_symbol_list,
                    for_ring=for_ring)))
        try:
            lhs = _add_ext_node(lhs, node_list)
        except ValueError:
            import pdb; pdb.set_trace()

    rhs = remove_tmp_edge(deepcopy(myself_hg))
    #rhs = remove_ext_node(rhs)
    #rhs = remove_nt_edge(rhs)
    try:
        rhs = _add_ext_node(rhs, node_list)
    except ValueError:
        import pdb; pdb.set_trace()

    nt_idx = 0
    if children_hg_list is not None:
        for each_child_hg in children_hg_list:
            node_list, edge_exists = common_node_list(myself_hg, each_child_hg)
            is_aromatic, _ = _check_aromatic(myself_hg, node_list)
            for_ring = _check_ring(each_child_hg)
            bond_symbol_list = []
            for each_node in node_list:
                bond_symbol_list.append(myself_hg.node_attr(each_node)['symbol'])
            rhs.add_edge(
                node_list,
                attr_dict=dict(
                    terminal=False,
                    nt_idx=nt_idx,
                    edge_exists=edge_exists,
                    symbol=NTSymbol(degree=len(node_list),
                                    is_aromatic=is_aromatic,
                                    bond_symbol_list=bond_symbol_list,
                                    for_ring=for_ring)))
            nt_idx += 1
    prod_rule = ProductionRule(lhs, rhs)
    prod_rule.subhg_idx = subhg_idx
    if DEBUG:
        if sorted(list(prod_rule.ext_node.keys())) \
           != list(np.arange(len(prod_rule.ext_node))):
            raise RuntimeError('ext_id is not continuous')
    return prod_rule


def remove_ext_node(hg):
    for each_node in hg.nodes:
        hg.node_attr(each_node).pop('ext_id', None)
    return hg

def remove_nt_edge(hg):
    remove_edge_list = []
    for each_edge in hg.edges:
        if not hg.edge_attr(each_edge)['terminal']:
            remove_edge_list.append(each_edge)
    hg.remove_edges(remove_edge_list)
    return hg

def remove_tmp_edge(hg):
    remove_edge_list = []
    for each_edge in hg.edges:
        if hg.edge_attr(each_edge).get('tmp', False):
            remove_edge_list.append(each_edge)
    hg.remove_edges(remove_edge_list)
    return hg
