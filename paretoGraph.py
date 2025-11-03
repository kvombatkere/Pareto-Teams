import time, json, pickle
from heapq import heappop, heappush, heapify
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import logging
logging.basicConfig(format='%(asctime)s |%(levelname)s: %(message)s', level=logging.INFO)

class paretoGraph():
    '''
    Define a class for coverage cost for n experts and single task, with pairwise costs
    Use for sum of pairwise and diameter costs
    '''

    def __init__(self, task, n_experts, pairwise_costs, size_univ, budget):
        '''
        Initialize instance with n experts and single task
        Each expert and task consists of a list of skills
        ARGS:
            task        : task to be accomplished;
            n_experts   : list of n experts; each expert is a list of skills
            costs       : cost of each expert
            size_univ   : number of distinct skills in the universe
            budget      : knapsack budget
        '''
        self.task = task
        self.task_skills = set(task)

        self.experts = n_experts
        self.m, self.n = size_univ, len(self.experts)
        self.pairwise_costs = pairwise_costs
        self.B = budget
        logging.info("Initialized Pareto Coverage - Graph Cost Instance, Task:{}, Num Experts:{}, Budget={}".format(self.task, self.n, self.B))


    