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


# In[ ]:


def get_edge(j_num, c_num, j_avl, c_avl):
    edge_all = [[],[]]
    for j in range(j_num):
        for c in range(c_num):
            edge_all[0].append(j)
            edge_all[1].append(c+j_num)
            edge_all[1].append(j)
            edge_all[0].append(c+j_num)
    edge_all2 = [[],[]]
    for j in range(j_num):
        for c in range(c_num):
            edge_all2[0].append(j)
            edge_all2[1].append(c+j_num)
    
    edge_all, edge_all2 = torch.tensor(edge_all), torch.tensor(edge_all2)
    crane = torch.tensor([i for i in range(c_num)])
    subset1 = torch.cat([j_avl, crane+j_num], 0)
    subset2 = torch.cat([j_avl, c_avl+j_num], 0)
    
    return subgraph(subset1 , edge_all)[0], subgraph(subset2 , edge_all2)[0] 



class GNN_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_dim   = args["input_dim"] 
        self.hidden_dim  = args["hidden_dim"]
        self.head =  args["head"]        
        self.out_dim = 1 # args["action_out_dim"]
        
        self.n_edr_1 = nn.Sequential(            
            nn.Linear(self.input_dim[0], self.hidden_dim, nn.ReLU())
        )
        self.n_edr_2 = nn.Sequential(            
            nn.Linear(self.input_dim[1], self.hidden_dim, nn.ReLU())
        )
        
        self.model_list = nn.ModuleList()
        for _ in range(3):
            self.model_list.append(                    GATv2Conv(in_channels = self.hidden_dim,                               out_channels = self.hidden_dim//self.head,                               heads = self.head, dropout = 0.2))
        self.edge_decoder = nn.Sequential(            
            nn.Linear(2*self.hidden_dim, self.hidden_dim), nn.ReLU(), 
            nn.Linear(self.hidden_dim  , self.hidden_dim), nn.ReLU(), 
            nn.Linear(self.hidden_dim  , self.out_dim),
        )
        
    def forward(self, state):
        ''' Note: 급하게 짜느라 우선 여기에 다 때려 넣었습니다... '''
        n_ft_1, n_ft_2 = state # (list,list) job, crane
        n_ft_1 = torch.tensor(n_ft_1).float().to(device)
        n_ft_2 = torch.tensor(n_ft_2).float().to(device)
        
        j_avl = torch.where(n_ft_1[:,0] == 0)[0]
        c_avl =  torch.where(n_ft_2[:,0] == 0)[0]

        n_ft_1 = self.n_edr_1(n_ft_1)
        n_ft_2 = self.n_edr_2(n_ft_2)
        n_ft = torch.cat([n_ft_1, n_ft_2], 0)
        
        j_num, c_num = n_ft_1.shape[0], n_ft_2.shape[0] 
        edge_1, edge_2 = get_edge(j_num, c_num, j_avl, c_avl)
        edge_1, edge_2 = edge_1.to(device), edge_2.to(device)
        for k in range(len(self.model_list)):
            n_ft = n_ft + self.model_list[k](n_ft, edge_1) 
        src, dst = edge_2
        # edge_feat = torch.cat([n_ft[src], n_ft[dst], j_ft, c_ft], dim=1)         
        
        
        edge_feat = torch.cat([n_ft[src], n_ft[dst]], dim=1)
        out = self.edge_decoder(edge_feat)
        return out, edge_2
    
    def getPolicy(self, state):  # get π(.|s), current policy given state
        logit, edge = self.forward(state)
        prob = F.softmax(logit.reshape(-1), -1)
        policy = Categorical(prob)
        return policy, edge

    def getAction(self, state):  # sample action following current policy
        policy, edge  = self.getPolicy(state)
        action = policy.sample()
        log_prob = policy.log_prob(action)
        edge_from = edge[0][action.item()].item()
        edge_to  = edge[1][action.item()].item() 
        return [edge_from,edge_to], log_prob.item()

    def get_max_action(self, state):
        policy, edge = self.getPolicy(state)
        action_probs = policy.probs
        best_action_idx = torch.argmax(action_probs)
        edge_from = edge[0][best_action_idx.item()].item() 
        edge_to  = edge[1][best_action_idx.item()].item() 
        return [edge_from,edge_to]
    
    def get_prob(self, state, action):  # sample action following current policy
        policy, edge  = self.getPolicy(state)  
        for i in range(len(edge[0])):
            if action == [edge[0][i].item(), edge[1][i].item()]:  
                act = i   
                break
        log_prob = policy.log_prob(torch.tensor(act)) 
        return log_prob 

