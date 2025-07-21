import time, json, pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import logging
logging.basicConfig(format='%(asctime)s |%(levelname)s: %(message)s', level=logging.INFO)

class paretoCoverageCost():

    def __init__(self, task, n_experts, costs, size_univ):
        '''
        Initialize instance with n experts and single task
        Each expert and task consists of a list of skills
        ARGS:
            task        : task to be accomplished;
            n_experts   : list of n experts; each expert is a list of skills
            costs       : cost of each expert
            size_univ   : number of distinct skills in the universe
        '''
        self.task = task
        self.experts = n_experts
        self.m, self.n = size_univ, len(self.experts)
        self.costs = costs
        logging.info("Initialized Pareto Coverage Cost Instance, Num Experts: {}".format(self.n))


    def getExpertCoverageAdd(self, cov_x, expert_index, curr_solution, curr_coverage):
        '''
        Helper function to get utility addition of new expert as per Demaine and Zadimoghaddam 2010
        '''
        expert_cov_add = len(curr_solution.union(self.experts[expert_index]).intersection(self.task))/len(self.task)
        expert_ratio_add = min(cov_x, expert_cov_add) - (curr_coverage/self.costs[expert_index])
        return expert_ratio_add


    def submodularWithBudget(self, cov_x, cost_B, epsilon_val):
        '''
        Greedy submodular maximization algorithm with knapsack budget from Demaine and Zadimoghaddam 2010
            If there exists an optimal solution with cost at most B and utility at least x, there is polytime
            algorithm that can find a collection of subsets of cost at most O(B log (1/eps)),
            and utility at least (1 - eps) x for any 0 < epsilon < 1
        ARGS:
            cov_x   : minimum desired coverage bound
            cost_B  : total knapsack cost of experts
        '''
        logging.info("Initializing")

        solution_skills = set()
        solution_expert_list = []
        curr_coverage, curr_cost = 0, 0

        while curr_coverage < ((1 - epsilon_val)*cov_x):
            expert_max_ratio = 0
            best_expert = None

            #Check all experts, only consider those not in solution
            for i, S_i in enumerate(self.experts):
                if S_i not in solution_expert_list:
                    expert_ratio = self.getExpertCoverageAdd(cov_x, i, solution_skills, curr_coverage)

                    if expert_ratio > expert_max_ratio:
                        best_expert = S_i
                        best_expert_cost = self.costs[i]
                        expert_max_ratio = expert_ratio

            #Add best expert to solution_skills and solution_expert_list
            solution_skills = solution_skills.union(set(best_expert))
            curr_coverage = len(solution_skills.intersection(self.task))/len(self.task)
            solution_expert_list.append(best_expert)
            curr_cost += best_expert_cost
            logging.info("Add expert: {} to solution, curr_coverage: {:.3f}, curr_cost: {}".format(best_expert, curr_coverage, curr_cost))

        return solution_expert_list