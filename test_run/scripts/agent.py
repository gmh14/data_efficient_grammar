import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class Agent(nn.Module):
    def __init__(self, feat_dim, hidden_size):
        super(Agent, self).__init__()
        self.affine1 = nn.Linear(feat_dim + 2, hidden_size)
        self.dropout = nn.Dropout(p=0.5)
        self.affine2 = nn.Linear(hidden_size, 2)
        self.saved_log_probs = {}

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        scores = self.affine2(x)
        return F.softmax(scores, dim=1)


def sample(agent, subgraph_feature, iter_num, sample_number):
    # subgraph_feature: N * (2+feat_dim), N is the number of subgraphs inside all inputs
    prob = agent(subgraph_feature)
    m = Categorical(prob)
    a = m.sample()
    take_action = (np.sum(a.numpy()) != 0)
    if take_action:
        if sample_number not in agent.saved_log_probs.keys():
            agent.saved_log_probs[sample_number] = {}
        if iter_num not in agent.saved_log_probs[sample_number].keys():
            agent.saved_log_probs[sample_number][iter_num] = [m.log_prob(a)]
        else:
            agent.saved_log_probs[sample_number][iter_num].append(m.log_prob(a))
    return a.numpy(), take_action
    
