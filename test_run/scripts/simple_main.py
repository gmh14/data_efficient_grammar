from rdkit import Chem
from copy import deepcopy
import numpy as np
from private import *
from grammar_generation import *
from agent import Agent
import torch.optim as optim
import torch.multiprocessing as mp
import logging
import torch
import math
import os
import time
import pprint
import pickle
import argparse
import fcntl
from retro_star_listener import lock


def evaluate(grammar, args, metrics=['diversity', 'syn']):
    # Metric evalution for the given gramamr
    div = InternalDiversity()
    eval_metrics = {}
    generated_samples = []
    generated_samples_canonical_sml = []
    iter_num_list = []
    idx = 0
    no_newly_generated_iter = 0
    print("Start grammar evaluation...")
    while(True):
        print("Generating sample {}/{}".format(idx, args.num_generated_samples))
        mol, iter_num = random_produce(grammar)
        if mol is None:
            no_newly_generated_iter += 1
            continue
        can_sml_mol = Chem.CanonSmiles(Chem.MolToSmiles(mol))
        if can_sml_mol not in generated_samples_canonical_sml:
            generated_samples.append(mol)
            generated_samples_canonical_sml.append(can_sml_mol)
            iter_num_list.append(iter_num)
            idx += 1
            no_newly_generated_iter = 0
        else:
            no_newly_generated_iter += 1
        if idx >= args.num_generated_samples or no_newly_generated_iter > 10:
            break

    for _metric in metrics:
        assert _metric in ['diversity', 'num_rules', 'num_samples', 'syn']
        if _metric == 'diversity':
            diversity = div.get_diversity(generated_samples)
            eval_metrics[_metric] = diversity
        elif _metric == 'num_rules':
            eval_metrics[_metric] = grammar.num_prod_rule
        elif _metric == 'num_samples':
            eval_metrics[_metric] = idx
        elif _metric == 'syn':
            eval_metrics[_metric] = retro_sender(generated_samples, args)
        else:
            raise NotImplementedError
    return eval_metrics


def retro_sender(generated_samples, args):
    # File communication to obtain retro-synthesis rate
    with open(args.receiver_file, 'w') as fw:
        fw.write('')
    while(True):
        with open(args.sender_file, 'r') as fr:
            editable = lock(fr)
            if editable:
                with open(args.sender_file, 'w') as fw:
                    for sample in generated_samples:
                        fw.write('{}\n'.format(Chem.MolToSmiles(sample)))
                break
            fcntl.flock(fr, fcntl.LOCK_UN)
    num_samples = len(generated_samples)
    print("Waiting for retro_star evaluation...")
    while(True):
        with open(args.receiver_file, 'r') as fr:
            editable = lock(fr)
            if editable:
                syn_status = []
                lines = fr.readlines()
                if len(lines) == num_samples:
                    for idx, line in enumerate(lines):
                        splitted_line = line.strip().split()
                        syn_status.append((idx, splitted_line[2]))
                    break
            fcntl.flock(fr, fcntl.LOCK_UN)
        time.sleep(1)
    assert len(generated_samples) == len(syn_status)
    return np.mean([int(eval(s[1])) for s in syn_status])


def learn(smiles_list, args):
    # Create logger
    save_log_path = 'log-num_generated_samples{}-{}'.format(args.num_generated_samples, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(save_log_path, scripts_to_save=[f for f in os.listdir('./') if f.endswith('.py')])
    logger = create_logger('global_logger', save_log_path + '/log.txt')
    logger.info('args:{}'.format(pprint.pformat(args)))
    logger = logging.getLogger('global_logger')

    # Initialize dataset & potential function (agent) & optimizer
    subgraph_set_init, input_graphs_dict_init = data_processing(smiles_list, args.GNN_model_path, args.motif)
    agent = Agent(feat_dim=300, hidden_size=args.hidden_size)
    if args.resume:
        assert  os.path.exists(args.resume_path), "Please provide valid path for resuming."
        ckpt = torch.load(args.resume_path)
        agent.load_state_dict(ckpt)

    # Start training
    logger.info('starting\n')
    curr_max_R = 0
    for train_epoch in range(args.max_epoches):
        returns = []
        log_returns = []
        logger.info("<<<<< Epoch {}/{} >>>>>>".format(train_epoch, args.max_epoches))

        # MCMC sampling
        for num in range(args.MCMC_size):
            grammar_init = ProductionRuleCorpus()
            l_input_graphs_dict = deepcopy(input_graphs_dict_init)
            l_subgraph_set = deepcopy(subgraph_set_init)
            l_grammar = deepcopy(grammar_init)
            iter_num, l_grammar, l_input_graphs_dict = MCMC_sampling(agent, l_input_graphs_dict, l_subgraph_set, l_grammar, num, args)
            # Grammar evaluation
            eval_metric = evaluate(l_grammar, args, metrics=['diversity', 'syn'])
            logger.info("eval_metrics: {}".format(eval_metric))
            # Record metrics
            R = eval_metric['diversity'] + 2 * eval_metric['syn']
            R_ind = R.copy()
            returns.append(R)
            log_returns.append(eval_metric)
            logger.info("======Sample {} returns {}=======:".format(num, R_ind))
            # Save ckpt
            if R_ind > curr_max_R:
                torch.save(agent.state_dict(), os.path.join(save_log_path, 'epoch_agent_{}_{}.pkl'.format(train_epoch, R_ind)))
                with open('{}/epoch_grammar_{}_{}.pkl'.format(save_log_path, train_epoch, R_ind), 'wb') as outp:
                    pickle.dump(l_grammar, outp, pickle.HIGHEST_PROTOCOL)
                with open('{}/epoch_input_graphs_{}_{}.pkl'.format(save_log_path, train_epoch, R_ind), 'wb') as outp:
                    pickle.dump(l_input_graphs_dict, outp, pickle.HIGHEST_PROTOCOL)
                curr_max_R = R_ind
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MCMC training')
    parser.add_argument('--training_data', type=str, default="./datasets/isocyanates.txt", help="file name of the training data")
    parser.add_argument('--GNN_model_path', type=str, default="./GCN/model_gin/supervised_contextpred.pth", help="file name of the pretrained GNN model")
    parser.add_argument('--hidden_size', type=int, default=128, help="hidden size of the potential function")
    parser.add_argument('--max_epoches', type=int, default=1, help="maximal training epoches")
    parser.add_argument('--num_generated_samples', type=int, default=100, help="number of generated samples to evaluate grammar")
    parser.add_argument('--MCMC_size', type=int, default=1, help="sample number of each step of MCMC")
    parser.add_argument('--learning_rate', type=int, default=1e-2, help="learning rate")
    parser.add_argument('--gammar', type=float, default=0.99, help="discount factor")
    parser.add_argument('--motif', action="store_true", default=False, help="use motif as the basic building block for polymer dataset")
    parser.add_argument('--sender_file', type=str, default="generated_samples.txt", help="file name of the generated samples")
    parser.add_argument('--receiver_file', type=str, default="output_syn.txt", help="file name of the output file of Retro*")
    parser.add_argument('--resume', action="store_true", default=False, help="resume model")
    parser.add_argument('--resume_path', type=str, default='', help="resume path")
    args = parser.parse_args()

    # Get raw training data
    assert os.path.exists(args.training_data), "Please provide valid path of training data."
    # Remove duplicated molecules
    with open(args.training_data, 'r') as fr:
        lines = fr.readlines()
        mol_sml = []
        for line in lines:
            if not (line.strip() in mol_sml):
                mol_sml.append(line.strip())

    # Clear the communication files for Retro*
    with open(args.sender_file, 'w') as fw:
        fw.write('')
    with open(args.receiver_file, 'w') as fw:
        fw.write('')
    
    # Grammar learning
    learn(mol_sml, args)

