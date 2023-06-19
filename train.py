#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import numpy as np
import pandas as pd

import torch 
from torch.distributions import Categorical
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from env import Simulator

import numpy as np 
import torch.nn as nn
import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.utils import subgraph

from crane_model import * 


# In[ ]:


def load_data(size, problem):
    data = f'./interval13/c2/n{size}/{problem}.pkl'
    with open(data, 'rb') as f:
        instance = pickle.load(f)
    return instance

class Trajectory(object):
    #  Note: batch 학습을  시키면 되는데 
    # 1) Multiprocess --> 이거 괜찮은데.... 
    # 2) env를 batch로 돌아가게 like Matnet
    
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.batch_state_list  = [[] for _ in range(batch_size)] 
        self.batch_action_list = [[] for _ in range(batch_size)]
        self.batch_reward_list = [[] for _ in range(batch_size)]
        
    # store 1 transition with log π(a|s)
    def push(self, batch, s, a, r):
        self.batch_state_list[batch].append(s)
        self.batch_action_list[batch].append(a)
        if pd.isnull(r):
            self.batch_reward_list[batch].append(0)
        else:
            self.batch_reward_list[batch].append(r)
    # 
    def compute_total_reward(self):
        self.total_reward = [[] for _ in range(self.batch_size)] 
        for batch in range(self.batch_size):
            epi_len = len(self.batch_reward_list[batch])
            for k in range(epi_len):
                self.total_reward[batch].append(                    sum(self.batch_reward_list[batch][k:]))
        self.total_reward = np.array(self.total_reward)
        return self.total_reward
    # 
    def compute_baseline(self):
        # self.compute_total_reward() # total_reward.shape : (batch, epi-len)
        baseline_list = []
        for k in range(self.total_reward.shape[1]):
            # print(self.total_reward)
            baseline_list.append(                np.sum(self.total_reward[:, k])/self.batch_size)
        return baseline_list
    
    def get_s_a(self, i, k):
        ''' '''
        state = self.batch_state_list[i][k]
        action = self.batch_action_list[i][k]
        return state, action
    
    
def run(args):
    batch_size = args['batch']
    learning_rate = args['lr']
    iter_num = args['iter']
    prob_size = args['prob_size']
    path = args['path']


    print("learning_rate", learning_rate)
    best_reward = 1e10
    policy_optimizer = Adam(model.parameters(), lr= learning_rate)
    for problem in range(iter_num):
        instance = load_data(prob_size, problem)
        memory = Trajectory(batch_size)
        model.train()
        for b_n in range(batch_size):
            env = Simulator(instance)
            for key in env.crane_lag.keys():
                if env.crane_lag[key] < -1e10:
                    env.crane_lag[key] = -100
            
            done = False
            s = env.get_state()
            while done == False:
                a, log_p, edge = model.getAction(s, env.crane_lag)
                action = [edge[0][a].item(), edge[1][a].item()- prob_size]
                sp, r, done = env.step(action)
                memory.push(b_n, s, a, r)
                s = sp
                
        total_reward  = memory.compute_total_reward()
        baseline_list = memory.compute_baseline()
        loss_1 = 0
        policy_optimizer.zero_grad()
        for k in range(len(baseline_list)):
            b_k = baseline_list[k]
            for b_n in range(batch_size):
                s,  a = memory.get_s_a(b_n, k)
                r_i_k = total_reward[b_n, k]
                loss = - model.get_prob(s, a, env.crane_lag) * (r_i_k -  b_k)
                loss.backward()
                loss_1 += loss.item()
        clip_grad_norm_(model.parameters(), 1.0)
        policy_optimizer.step()

        model.eval()
        r_list = []
        for eval_prob in range(100):
            instance = load_data(20 , eval_prob)
            env = Simulator(instance)
            for key in env.crane_lag.keys():
                if env.crane_lag[key] < -1e10:
                    env.crane_lag[key] = -100
            
            r_list.append([])
            done = False
            s = env.get_state()
            while done == False:
                a, edge = model.get_max_action(s, env.crane_lag)
                action = [edge[0][a].item(), edge[1][a].item()- 20]
                s, r, done = env.step(action)
            r_list[-1].append(env.obj_value)
        print(f'iter {problem}: mean_TCT: {np.sum(r_list)}')
    
        if best_reward > np.sum(r_list):
            best_reward = np.sum(r_list)
            torch.save(model.state_dict(), path)
        
args = {
    'prob_size' : 20 ,
    # train_params  
    'epoch': 100 , 
    'iter': 100 , 
    
    'lr':  2.5e-3,
    'batch': 16,
    'tolerance':  3,
    
    # model_params  
    'input_dim':[4, 6], 
    'hidden_dim': 16, 
    'head' : 8, 
    'model': 'GIN', # 'GNN', 'GIN', 'GATv2', 'grpahormer'
    'device':'cuda' if torch.cuda.is_available() else 'cpu' , 
    "path": "model_name.pt"
}



if __name__ == '__main__':
    device = args['device']
    model = GNN_Model(args).to(device)
    run(args)

