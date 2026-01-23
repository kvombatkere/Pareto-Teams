import time, pickle
from heapq import heappop, heappush, heapify
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import logging
logging.basicConfig(format='%(asctime)s |%(levelname)s: %(message)s', level=logging.INFO)

class paretoCardinalityTeams():
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
                logging.debug("k = {}, Adding expert {}, curr_coverage={:.3f}".format(k_val, self.experts[top_expert_indx], curr_coverage))
        
            #Otherwise re-insert top expert into heap with updated marginal gain
            else:
                updated_top_expert = (top_expert_marginal_gain*-1, top_expert_indx)
                heappush(self.maxHeap, updated_top_expert)

        runTime = time.perf_counter() - startTime
        logging.info("Cardinality Greedy Solution for k_max:{}, Coverage:{:.3f}, Runtime = {:.2f} seconds".format(solution_experts, curr_coverage, runTime))

        return solution_experts, solution_skills, curr_coverage, runTime
    
    def top_k(self):
        '''
        Top-k Algorithm: Select the top k experts from the heap without updates.
        Marginal gains are computed w.r.t. the empty set (i.e., individual coverages).
        '''
        startTime = time.perf_counter()

        #Solution skills and experts
        solution_skills = set()
        solution_experts = []

        #Create maxheap with coverages
        self.createExpertCoverageMaxHeap()

        #Select top k experts
        for k_val in range(1, self.k_max + 1):
            if self.maxHeap:
                top_expert_key = heappop(self.maxHeap)
                top_expert_indx = top_expert_key[1]
                top_expert_skills = set(self.experts[top_expert_indx])

                solution_skills = solution_skills.union(top_expert_skills)
                solution_experts.append(self.experts[top_expert_indx])
                curr_coverage = len(solution_skills.intersection(self.task_skills)) / len(self.task)
                self.kSolDict[k_val] = {"Experts": solution_experts.copy(), "Skills": solution_skills.copy(), "Coverage": curr_coverage}
                logging.debug("k = {}, Adding top expert {}, curr_coverage={:.3f}".format(k_val, self.experts[top_expert_indx], curr_coverage))

        runTime = time.perf_counter() - startTime
        logging.info("Top-k Solution for k_max:{}, Coverage:{:.3f}, Runtime = {:.2f} seconds".format(solution_experts, curr_coverage, runTime))

        return solution_experts, solution_skills, curr_coverage, runTime


    def random_selection(self):
        '''
        Random Algorithm: Randomly select k distinct experts.
        '''
        startTime = time.perf_counter()

        #Randomly select k_max distinct expert indices
        selected_indices = np.random.choice(self.n, size=self.k_max, replace=False)

        #Solution skills and experts
        solution_skills = set()
        solution_experts = []

        for idx in selected_indices:
            expert_skills = set(self.experts[idx])
            solution_skills = solution_skills.union(expert_skills)
            solution_experts.append(self.experts[idx])

        curr_coverage = len(solution_skills.intersection(self.task_skills)) / len(self.task)

        #Populate kSolDict for consistency
        for k_val in range(1, self.k_max + 1):
            partial_experts = solution_experts[:k_val]
            partial_skills = set()
            for exp in partial_experts:
                partial_skills = partial_skills.union(set(exp))
            partial_coverage = len(partial_skills.intersection(self.task_skills)) / len(self.task)
            self.kSolDict[k_val] = {"Experts": partial_experts, "Skills": partial_skills, "Coverage": partial_coverage}

        runTime = time.perf_counter() - startTime
        logging.info("Random Selection Solution for k_max:{}, Coverage:{:.3f}, Runtime = {:.2f} seconds".format(solution_experts, curr_coverage, runTime))

        return solution_experts, solution_skills, curr_coverage, runTime
    

def import_pickled_datasets(dataset_name, dataset_num):
    '''
    Code to quickly import final datasets for experiments
    '''
    data_path = '../../datasets/pickled_data/' + dataset_name + '/' + dataset_name + '_'
    
    #Import pickled data
    with open(data_path + 'experts_{}.pkl'.format(dataset_num), "rb") as fp:
        experts = pickle.load(fp)
        logging.info("Imported {} experts, Num Experts: {}".format(dataset_name, len(experts)))

    with open(data_path + 'tasks_{}.pkl'.format(dataset_num), "rb") as fp:
        tasks = pickle.load(fp)
        logging.info("Imported {} tasks, Num Tasks: {}".format(dataset_name, len(tasks)))

    with open(data_path + 'costs_{}.pkl'.format(dataset_num), "rb") as fp:
        costs_arr = pickle.load(fp)
        logging.info("Imported {} costs, Num Costs: {}".format(dataset_name, len(costs_arr)))

    with open(data_path + 'graphMat_{}.pkl'.format(dataset_num), "rb") as fp:
        graphmat = pickle.load(fp)
        logging.info("Imported {} graph matrix, Shape: {}\n".format(dataset_name, graphmat.shape))

    return experts, tasks, costs_arr, graphmat