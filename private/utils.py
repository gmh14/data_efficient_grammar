from copy import deepcopy
from typing import List
import numpy as np
import os
import logging
import shutil


def _node_match(node1, node2):
    # if the nodes are hyperedges, `atom_attr` determines the match
    if node1['bipartite'] == 'edge' and node2['bipartite'] == 'edge':
        return node1["attr_dict"]['symbol'] == node2["attr_dict"]['symbol']
    elif node1['bipartite'] == 'node' and node2['bipartite'] == 'node':
        # bond_symbol
        return node1['attr_dict']['symbol'] == node2['attr_dict']['symbol']
    else:
        return False

def _easy_node_match(node1, node2):
    # if the nodes are hyperedges, `atom_attr` determines the match
    if node1['bipartite'] == 'edge' and node2['bipartite'] == 'edge':
        return node1["attr_dict"].get('symbol', None) == node2["attr_dict"].get('symbol', None)
    elif node1['bipartite'] == 'node' and node2['bipartite'] == 'node':
        # bond_symbol
        return node1['attr_dict'].get('ext_id', -1) == node2['attr_dict'].get('ext_id', -1)\
            and node1['attr_dict']['symbol'] == node2['attr_dict']['symbol']
    else:
        return False


def _node_match_prod_rule(node1, node2, ignore_order=False):
    # if the nodes are hyperedges, `atom_attr` determines the match
    if node1['bipartite'] == 'edge' and node2['bipartite'] == 'edge':
        return node1["attr_dict"]['symbol'] == node2["attr_dict"]['symbol']
    elif node1['bipartite'] == 'node' and node2['bipartite'] == 'node':
        # ext_id, order4hrg, bond_symbol
        if ignore_order:
            return node1['attr_dict']['symbol'] == node2['attr_dict']['symbol']
        else:
            return node1['attr_dict']['symbol'] == node2['attr_dict']['symbol']\
                and node1['attr_dict'].get('ext_id', -1) == node2['attr_dict'].get('ext_id', -1)
    else:
        return False


def _edge_match(edge1, edge2, ignore_order=False):
    #return True
    if ignore_order:
        return True
    else:
        return edge1["order"] == edge2["order"]

def masked_softmax(logit, mask):
    ''' compute a probability distribution from logit

    Parameters
    ----------
    logit : array-like, length D
        each element indicates how each dimension is likely to be chosen
        (the larger, the more likely)
    mask : array-like, length D
        each element is either 0 or 1.
        if 0, the dimension is ignored
        when computing the probability distribution.

    Returns
    -------
    prob_dist : array, length D
        probability distribution computed from logit.
        if `mask[d] = 0`, `prob_dist[d] = 0`.
    '''
    if logit.shape != mask.shape:
        raise ValueError('logit and mask must have the same shape')
    c = np.max(logit)
    exp_logit = np.exp(logit - c) * mask
    sum_exp_logit = exp_logit @ mask
    return exp_logit / sum_exp_logit


def create_logger(name, log_file, level=logging.INFO):
    l = logging.getLogger(name)
    formatter = logging.Formatter(
        '[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    l.addHandler(sh)
    return l

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
