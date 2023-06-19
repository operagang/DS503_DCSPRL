#!/usr/bin/env python
# coding: utf-8



import pickle
import numpy as np

import pandas as pd
from env import Simulator

import torch 
from torch.distributions import Categorical
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_


def load_data(size, problem):
    data = f'./interval13/c2/n{size}/{problem}.pkl'
    with open(data, 'rb') as f:
        instance = pickle.load(f)
    return instance

def first_come_nearst(self):
    possible_tasks =  []
    best_time = 1e10
    for task in range(len(self.task_dict)):
        tmp = self.task_dict[task]
        if tmp.arr <= self.clock and task not in self.assigned_tasks:
            possible_tasks.append(task)
            if tmp.arr < best_time:
                best_task = task
                best_time = tmp.arr 

    nearliest_time = 1e10
    possible_crane = []
    for crane in range(len(self.crane_dict)):
        tmp = self.crane_dict[crane]
        if tmp.assigned == False:
            possible_crane.append(crane)
            distance = np.absolute(self.task_dict[best_task].s - tmp.pos)
            if  distance < nearliest_time:
                nearliest_time = distance
                best_crane = crane 
                
    return [best_task, best_crane]
## 
def nearst(self):
    possible_tasks =  []
    best_time = 1e10
    nearliest_time = 1e10
    for task in range(len(self.task_dict)):
        tmp = self.task_dict[task]
        if tmp.arr <= self.clock and task not in self.assigned_tasks:
            possible_tasks.append(task)
            # if tmp.arr < best_time:
            #    best_task = task
            #    best_time = tmp.arr 
            possible_crane = []
            for crane in range(len(self.crane_dict)):
                tmp = self.crane_dict[crane]
                if tmp.assigned == False:
                    possible_crane.append(crane)
                    distance = np.absolute(self.task_dict[task].s - tmp.pos)
                    if  distance < nearliest_time:
                        nearliest_time = distance
                        best_crane = crane 
                        best_task = task
    return [best_task, best_crane]


def run(args):
    iter_num = args['iter']
    prob_size = args['prob_size']
    option = args['option']
    obj_list = []
    for problem in range(iter_num):
        instance = load_data(prob_size, problem)
        env = Simulator(instance)
        done = False
        while done == False:
            if option == 'first_come_nearst':
                a = first_come_nearst(env)
            elif option == 'nearst':
                a = nearst(env)
            elif option == 'STD':
                a = STD(env)
            sp, r, done = env.step(a)
        obj_list.append(env.obj_value)
    return obj_list


if __name__ == '__main__':
    args = {
        'prob_size' : 30 ,
        'iter': 100 , 
        'option' : 'first_come_nearst' # 
    }
    print("option: first_come_nearst")
    for eval_prob_size in range(2, 11):
        args['prob_size'] = eval_prob_size*10
        obj = run(args)
        print(f'{eval_prob_size*10}: {np.mean(obj)}')

    args = {
        'prob_size' : 30 ,
        'iter': 100 , 
        'option' : 'nearst' # 
    }
    print()
    print("option: MinDistance")
    for eval_prob_size in range(2, 11):
        args['prob_size'] = eval_prob_size*10
        obj = run(args)
        print(f'{eval_prob_size*10}: {np.mean(obj)}')

