#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
import pandas as pd

import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.utils import subgraph

from env import Simulator
from crane_model import * 



def load_data(size, problem):
    data = f'./interval13/c2/n{size}/{problem}.pkl'
    with open(data, 'rb') as f:
        instance = pickle.load(f)
    return instance




def eval(args):
    batch_size = args['batch']
    learning_rate = args['lr']
    iter_num = args['iter']
    prob_size = args['prob_size']
    eval_prob_size = args['eval_prob_size']
    
    model.eval()
    r_list = []
    for eval_prob in range(100):
        instance = load_data(eval_prob_size , eval_prob)
        env = Simulator(instance)
        for key in env.crane_lag.keys():
            if env.crane_lag[key] < -1e10:
                env.crane_lag[key] = -100
        
        r_list.append([])
        done = False
        s = env.get_state()
        while done == False:
            a, edge = model.get_max_action(s, env.crane_lag)
            action = [edge[0][a].item(), edge[1][a].item()- eval_prob_size]
            s, r, done = env.step(action)
        r_list[-1].append(env.obj_value)
    print(f'{eval_prob_size}: {np.sum(r_list)}')


args = {
    'prob_size' : 20 ,
    # train_params  
    'epoch': 100 , 
    'iter': 100 , 
    
    'lr':  2.5e-4,
    'batch': 16,
    'tolerance':  3,
    
    # model_params  
    'input_dim':[4, 6], 
    'hidden_dim': 16, 
    'head' : 8, 
    'model': 'GIN', # 'GNN', 'GIN', 'GATv2', 'grpahormer'
     
    # model  
    "device": 'cuda' if torch.cuda.is_available() else 'cpu' , 
    "PATH": "GNN_v2_not_fully0.002516.pt" # "GNN_v2_not_fully0.00516.pt"
}


if __name__ == '__main__':
    model = GNN_Model(args).to(args['device'])
    model.load_state_dict(torch.load(args['PATH']))
    for size in [30,40,50,60,70,80,90,100]:
        args['eval_prob_size'] = size
        eval(args)    




