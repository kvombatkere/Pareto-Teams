import time, json, pickle
from heapq import heappop, heappush, heapify
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import logging
logging.basicConfig(format='%(asctime)s |%(levelname)s: %(message)s', level=logging.INFO)

class paretoCardinality():
    '''
    Define a class for coverage cost for n experts and single task, with cardinality cost
    '''

    def __init__(self, task, n_experts, size_univ, k_max):
        '''
        Initialize instance with n experts and single task
        Each expert and task consists of a list of skills
        ARGS:
            task        : task to be accomplished;
            n_experts   : list of n experts; each expert is a list of skills
            k           : cardinality constraint
            size_univ   : number of distinct skills in the universe
        '''
        self.task = task
        self.task_skills = set(task)

        self.experts = n_experts
        self.m, self.n = size_univ, len(self.experts)
        self.k_max = k_max
        self.kSolDict = {}
        logging.info("Initialized Pareto Coverage - Cardinality Cost Instance, Task:{}, Num Experts:{}, k={}".format(self.task, self.n, k_max))


    def createExpertCoverageMaxHeap(self):
        '''
        Initialize self.maxHeap with expert-task coverages for each expert
        '''
        #Create max heap to store edge coverags
        self.maxHeap = []
        heapify(self.maxHeap)
        
        for i, E_i in enumerate(self.experts):
            expert_skills = set(E_i)

            #Compute expert-task coverage 
            expert_coverage = len(expert_skills.intersection(self.task_skills))/len(self.task)

            #push to maxheap - heapItem stored -gain, expert index and cost
            heapItem = (expert_coverage*-1, i)
            heappush(self.maxHeap, heapItem)

        return 


    def greedyCardinality(self):
        '''
        Greedy Algorithm for Submodular Maximization
        '''
        startTime = time.perf_counter()

        #Solution skills and experts
        solution_skills = set()
        solution_experts = [] 

        curr_coverage, coverage_list = 0, [0]
        k_val = 0

        #Create maxheap with coverages
        self.createExpertCoverageMaxHeap()

        #Assign experts greedily using max heap until the solution has size k_max or coverage <= 1
        while len(self.maxHeap) > 1 and (len(solution_experts) < self.k_max) and (curr_coverage < 1):
            
            #Pop best expert from maxHeap and compute marginal gain
            top_expert_key = heappop(self.maxHeap)
            top_expert_indx = top_expert_key[1]
            top_expert_skills = set(self.experts[top_expert_indx]) #Get the skills of the top expert

            sol_with_top_expert = solution_skills.union(top_expert_skills)
            coverage_with_top_expert = len(sol_with_top_expert.intersection(self.task_skills))/len(self.task)
            top_expert_marginal_gain = (coverage_with_top_expert - curr_coverage)

            #Check expert now on top - 2nd expert on heap
            second_expert = self.maxHeap[0] 
            second_expert_heap_gain = second_expert[0]*-1

            #If marginal gain of top expert is better we add to solution
            if top_expert_marginal_gain >= second_expert_heap_gain:
                k_val += 1
                solution_skills = solution_skills.union(top_expert_skills)
                solution_experts.append(self.experts[top_expert_indx])
                curr_coverage = coverage_with_top_expert
                self.kSolDict[k_val] = {"Experts": solution_experts, "Skills":solution_skills, "Coverage":curr_coverage}
                logging.info("k = {}, Adding expert {}, curr_coverage={:.3f}".format(k_val, self.experts[top_expert_indx], curr_coverage))
        
            #Otherwise re-insert top expert into heap with updated marginal gain
            else:
                updated_top_expert = (top_expert_marginal_gain*-1, top_expert_indx)
                heappush(self.maxHeap, updated_top_expert)

        runTime = time.perf_counter() - startTime
        logging.info("Cardinality Greedy Solution for k_max:{}, Coverage:{:.3f}, Runtime = {:.2f} seconds".format(solution_experts, curr_coverage, runTime))

        return solution_experts, solution_skills, curr_coverage, runTime