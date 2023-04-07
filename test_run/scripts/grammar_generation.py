from rdkit import Chem
from functools import partial
from multiprocessing import Pool
from fuseprop import find_clusters, extract_subgraph, get_mol, get_smiles, find_fragments
from copy import deepcopy
import numpy as np
import torch
import argparse
from private import *
from agent import sample


def data_processing(input_smiles, GNN_model_path, motif=False):
    input_mols = []
    input_graphs = []
    init_subgraphs = []
    subgraphs_idx = []
    input_graphs_dict = {}
    init_edge_flag = 0

    for n, smiles in enumerate(input_smiles):
        print("data processing {}/{}".format(n, len(input_smiles)))
        # Kekulized
        smiles = get_smiles(get_mol(smiles))
        mol = get_mol(smiles)
        input_mols.append(mol)
        if not motif:
            clusters, atom_cls = find_clusters(mol)
            for i,cls in enumerate(clusters):
                clusters[i] = set(list(cls))
            for a in range(len(atom_cls)):
                atom_cls[a] = set(atom_cls[a])
        else:
            fragments = find_fragments(mol)
            clusters = [frag[1] for frag in fragments]
        
        # Construct graphs
        subgraphs = []
        subgraphs_idx_i = [] 
        for i,cluster in enumerate(clusters):
            _, subgraph_i_mapped, _ = extract_subgraph(smiles, cluster)
            subgraphs.append(SubGraph(subgraph_i_mapped, mapping_to_input_mol=subgraph_i_mapped, subfrags=list(cluster)))
            subgraphs_idx_i.append(list(cluster))
            init_edge_flag += 1
        
        init_subgraphs.append(subgraphs)
        subgraphs_idx.append(subgraphs_idx_i)
        graph = InputGraph(mol, smiles, subgraphs, subgraphs_idx_i, GNN_model_path)
        input_graphs.append(graph)
        input_graphs_dict[MolKey(graph.mol)] = graph

    # Construct subgraph_set
    subgraph_set = SubGraphSet(init_subgraphs, subgraphs_idx, input_graphs)
    return subgraph_set, input_graphs_dict


def grammar_generation(agent, input_graphs_dict, subgraph_set, grammar, mcmc_iter, sample_number, args):
    # Selected hyperedge (subgraph)
    plist = [*subgraph_set.map_to_input]

    # Terminating condition
    if len(plist) == 0:
        # done_flag, new_input_graphs_dict, new_subgraph_set, new_grammar
        return True, input_graphs_dict, subgraph_set, grammar

    # Update every InputGraph: remove every subgraph that equals to p_star, for those subgraphs that contain atom idx in p_star, replace the atom with p_star
    org_input_graphs_dict = deepcopy(input_graphs_dict)
    org_subgraph_set = deepcopy(subgraph_set)
    org_grammar = deepcopy(grammar)

    input_graphs_dict = deepcopy(org_input_graphs_dict)
    subgraph_set = deepcopy(org_subgraph_set)
    grammar = deepcopy(org_grammar)

    for i, (key, input_g) in enumerate(input_graphs_dict.items()):
        print("---for graph {}---".format(i))
        action_list = []
        all_final_features = []
        # Skip the final iteration for training agent
        if len(input_g.subgraphs) > 1:
            for subgraph, subgraph_idx in zip(input_g.subgraphs, input_g.subgraphs_idx):
                subg_feature = input_g.get_subg_feature_for_agent(subgraph)
                num_occurance = subgraph_set.map_to_input[MolKey(subgraph)][1]
                num_in_input = len(subgraph_set.map_to_input[MolKey(subgraph)][0].keys())
                final_feature = []
                final_feature.extend(subg_feature.tolist())
                final_feature.append(1 - np.exp(-num_occurance))
                final_feature.append(num_in_input / len(list(input_graphs_dict.keys())))
                all_final_features.append(torch.unsqueeze(torch.from_numpy(np.array(final_feature)).float(), 0))
            while(True):
                action_list, take_action = sample(agent, torch.vstack(all_final_features), mcmc_iter, sample_number)
                if take_action:
                    break
        elif len(input_g.subgraphs) == 1:
            action_list = [1]
        else:
            continue
        print("Hyperedge sampling:", action_list)
        # Merge connected hyperedges
        p_star_list = input_g.merge_selected_subgraphs(action_list)
        # Generate rules
        for p_star in p_star_list:
            is_inside, subgraphs, subgraphs_idx = input_g.is_candidate_subgraph(p_star)
            if is_inside:
                for subg, subg_idx in zip(subgraphs, subgraphs_idx):
                    if subg_idx not in input_g.subgraphs_idx:
                        # Skip the subg if it has been merged in previous iterations
                        continue
                    grammar = generate_rule(input_g, subg, grammar)
                    input_g.update_subgraph(subg_idx)
                    
    # Update subgraph_set
    subgraph_set.update([g for (k, g) in input_graphs_dict.items()])
    new_grammar = deepcopy(grammar)
    new_input_graphs_dict = deepcopy(input_graphs_dict)
    new_subgraph_set = deepcopy(subgraph_set)
    return False, new_input_graphs_dict, new_subgraph_set, new_grammar


def MCMC_sampling(agent, all_input_graphs_dict, all_subgraph_set, all_grammar, sample_number, args):
    iter_num = 0
    while(True):
        print("======MCMC iter{}======".format(iter_num))
        done_flag, new_input_graphs_dict, new_subgraph_set, new_grammar = grammar_generation(agent, all_input_graphs_dict, all_subgraph_set, all_grammar, iter_num, sample_number, args)
        print("Graph contraction status: ", done_flag)
        if done_flag:
            break
        all_input_graphs_dict = deepcopy(new_input_graphs_dict)
        all_subgraph_set = deepcopy(new_subgraph_set)
        all_grammar = deepcopy(new_grammar)
        iter_num += 1

    return iter_num, new_grammar, new_input_graphs_dict


def random_produce(grammar):
    def sample(l, prob=None):
        if prob is None:
            prob = [1/len(l)] * len(l)
        idx =  np.random.choice(range(len(l)), 1, p=prob)[0]
        return l[idx], idx

    def prob_schedule(_iter, selected_idx):
        prob_list = []
        # prob = exp(a * t * x), x = {0, 1}
        a = 0.5
        for rule_i, rule in enumerate(grammar.prod_rule_list):
            x = rule.is_ending
            if rule.is_start_rule:
                prob_list.append(0)
            else:
                prob_list.append(np.exp(a * _iter * x))
        prob_list = np.array(prob_list)[selected_idx]
        prob_list = prob_list / np.sum(prob_list)
        return prob_list

    hypergraph = Hypergraph()
    starting_rules = [(rule_i, rule) for rule_i, rule in enumerate(grammar.prod_rule_list) if rule.is_start_rule]
    iter = 0
    while(True):
        if iter == 0:
            _, idx = sample(starting_rules)
            selected_rule_idx, selected_rule = starting_rules[idx]
            hg_cand, _, avail = selected_rule.graph_rule_applied_to(hypergraph)
            hypergraph = deepcopy(hg_cand)
        else:
            candidate_rule = []
            candidate_rule_idx = []
            candidate_hg = []
            for rule_i, rule in enumerate(grammar.prod_rule_list):
                hg_prev = deepcopy(hypergraph)
                hg_cand, _, avail = rule.graph_rule_applied_to(hypergraph)
                if(avail):
                    candidate_rule.append(rule)
                    candidate_rule_idx.append(rule_i)
                    candidate_hg.append(hg_cand)
            if (all([rl.is_start_rule for rl in candidate_rule]) and iter > 0) or iter > 30:
                break
            prob_list = prob_schedule(iter, candidate_rule_idx)
            hypergraph, idx = sample(candidate_hg, prob_list)
            selected_rule = candidate_rule_idx[idx]
        iter += 1
    try:
        mol = hg_to_mol(hypergraph)
        print(Chem.MolToSmiles(mol))
    except:
        return None, iter

    return mol, iter
