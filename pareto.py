import time, json, pickle
from heapq import heappop, heappush, heapify
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import logging
logging.basicConfig(format='%(asctime)s |%(levelname)s: %(message)s', level=logging.INFO)

class paretoCoverageCost():
    '''
    Define a class for coverage cost for n experts and single task
    '''

    def __init__(self, task, n_experts, costs, size_univ, budget):
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
        self.costs = costs
        self.B = budget
        logging.info("Initialized Pareto Coverage-Cost Instance, Task:{}, Num Experts:{}, Budget={}".format(self.task, self.n, self.B))


    def getExpertCoverageAdd(self, cov_x, expert_index, curr_solution, curr_coverage):
        '''
        Helper function to get utility addition of new expert as per Demaine and Zadimoghaddam 2010
        '''
        expert_cov_add = len(curr_solution.union(self.experts[expert_index]).intersection(self.task))/len(self.task)
        expert_ratio_add = (min(cov_x, expert_cov_add) - curr_coverage)/self.costs[expert_index]
        return expert_ratio_add


    def submodularWithBudget(self, cov_x, epsilon_val):
        '''
        Greedy submodular maximization algorithm with knapsack budget from Demaine and Zadimoghaddam 2010
            If there exists an optimal solution with cost at most B and utility at least x, there is polytime
            algorithm that can find a collection of subsets of cost at most O(B log (1/eps)),
            and utility at least (1 - eps) x for any 0 < epsilon < 1
        ARGS:
            cov_x   : minimum desired coverage bound
        RETURN:
            solution_expert_list    : List of chosen experts
        '''
        solution_skills = set()
        solution_expert_list = []
        curr_coverage, curr_cost = 0, 0
        coverage_list, cost_list = [0], [0]

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
            curr_coverage = len(solution_skills.intersection(self.task_skills))/len(self.task)
            solution_expert_list.append(best_expert)
            curr_cost += best_expert_cost
            logging.info("Added expert: {} to solution, curr_coverage: {:.3f}, curr_cost: {}".format(best_expert, curr_coverage, curr_cost))

            #Update incremental coverage and cost
            coverage_list.append(curr_coverage)
            cost_list.append(curr_cost)

        logging.info("Final solution: {}, coverage: {}, cost: {}".format(solution_expert_list, curr_coverage, curr_cost))
        self.plotParetoCurve(coverage_list, cost_list)

        return solution_expert_list
    

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
            expert_weight = expert_coverage/self.costs[i]

            #push to maxheap - heapItem stored -gain, expert index and cost
            heapItem = (expert_weight*-1, i, self.costs[i])
            heappush(self.maxHeap, heapItem)

        return 
    

    def plainGreedy(self):
        '''
        Adapt Plain Greedy Algorithm from  Feldman, Nutov, Shoham 2021; Practical Budgeted Submodular Maximization
        Run with input sets, self.experts instead of individual elements
        '''
        startTime = time.perf_counter()

        #Solution skills and experts
        solution_skills = set()
        solution_experts = [] 

        curr_coverage, curr_cost = 0, 0
        coverage_list, cost_list = [0], [0]

        #Create maxheap with coverages
        self.createExpertCoverageMaxHeap()

        #Assign experts greedily using max heap
        #Check if there is an element with cost that fits in budget
        while len(self.maxHeap) > 1 and (min(key[2] for key in self.maxHeap) <= (self.B - curr_cost)) and (curr_coverage < 1):
            
            #Pop best expert from maxHeap and compute marginal gain
            top_expert_key = heappop(self.maxHeap)
            top_expert_indx, top_expert_cost = top_expert_key[1], top_expert_key[2]
            top_expert_skills = set(self.experts[top_expert_indx]) #Get the skills of the top expert

            sol_with_top_expert = solution_skills.union(top_expert_skills)
            coverage_with_top_expert = len(sol_with_top_expert.intersection(self.task_skills))/len(self.task)
            top_expert_marginal_gain = (coverage_with_top_expert - curr_coverage)/top_expert_cost

            #Check expert now on top - 2nd expert on heap
            second_expert = self.maxHeap[0] 
            second_expert_heap_gain = second_expert[0]*-1

            #If marginal gain of top expert is better we add to solution
            if top_expert_marginal_gain >= second_expert_heap_gain:
                #Only add if expert is within budget
                if top_expert_cost + curr_cost <= self.B:
                    solution_skills = solution_skills.union(top_expert_skills)
                    solution_experts.append(self.experts[top_expert_indx])
                    curr_coverage = coverage_with_top_expert
                    curr_cost += top_expert_cost
                    logging.info("Adding expert {}, curr_coverage={:.3f}, curr_cost={}".format(self.experts[top_expert_indx], curr_coverage, curr_cost))
            
            #Otherwise re-insert top expert into heap with updated marginal gain
            else:
                updated_top_expert = (top_expert_marginal_gain*-1, top_expert_indx, top_expert_cost)
                heappush(self.maxHeap, updated_top_expert)

        runTime = time.perf_counter() - startTime
        logging.info("Plain Greedy Solution:{}, Coverage:{:.3f}, Cost:{}, Runtime = {:.2f} seconds".format(solution_experts, curr_coverage, curr_cost, runTime))

        return solution_experts, solution_skills, curr_coverage, curr_cost
    

    def greedyPlus(self):
        '''
        Greedy Plus Algorithm from  Feldman, Nutov, Shoham 2021; Practical Budgeted Submodular Maximization
        Greedy returns the better solution among the output of Plain Greedy and the best feasible solution 
        that can be obtained by combining any solution that Plain Greedy had at some iteration 
        with a single expert.
        '''
        startTime = time.perf_counter()

        #Get plain greedy solution
        sol_experts, sol_skills, best_coverage, best_cost = self.plainGreedy()

        logging.info("=="*50)
        best_experts_list, feasible_expert_list, feasible_expert_skills = [], [], set()
        feasible_expert_cost = 0

        #Loop over solution in each iteration of plain greedy
        for i, expert_i in enumerate(sol_experts):
            feasible_expert_list.append(expert_i)
            feasible_expert_skills = feasible_expert_skills.union(set(expert_i))
            feasible_expert_cost += self.costs[self.experts.index(expert_i)]
            logging.info("Trying incremental solution:{}, cost:{}".format(feasible_expert_list, feasible_expert_cost))
            
            for j, E_j in enumerate(self.experts):
                #If adding a single expert doesn't violate budget
                if feasible_expert_cost + self.costs[j] <= self.B:
                    #Compute coverage by adding expert to incremental solution
                    added_expert_cov = len((feasible_expert_skills.union(set(E_j))).intersection(self.task_skills))/len(self.task)
                    
                    #If this solution is better than original solution, store it
                    if added_expert_cov > best_coverage:
                        best_experts_list = feasible_expert_list.copy()
                        best_experts_list.append(E_j)
                        best_coverage = added_expert_cov
                        best_cost = feasible_expert_cost + self.costs[j]
                        logging.info("New feasible solution yielded better coverage! {}, coverage={:.3f}, cost={}".format(best_experts_list,best_coverage,best_cost))
        
        #Return original solution if that is better
        if len(best_experts_list) == 0:
            logging.info("Original Plain Greedy Solution was best!")
            best_experts_list = sol_experts

        runTime = time.perf_counter() - startTime
        logging.info("Greedy+ Solution:{}, Coverage:{:.3f}, Cost:{}, Runtime = {:.2f} seconds".format(best_experts_list, best_coverage, best_cost, runTime))
        
        #Return solution
        return best_experts_list, sol_skills, best_coverage, best_cost
    


    def createmaxHeap2Guess(self, expert_pair_key, expert_pair_data):
        '''
        Initialize self.maxHeap2Guess with expert-task coverages for each expert
        '''
        #Create max heap to store coverages with respect to new objective function
        self.maxHeap2Guess = []
        heapify(self.maxHeap2Guess)

        #Compute skills, cost and coverage of pair
        expertPairSkills, expertPairCost = expert_pair_data[0], expert_pair_data[1]
        expertPairCoverage = len(expertPairSkills.intersection(self.task_skills))/len(self.task)
        
        for i, E_i in enumerate(self.experts):
            if i not in expert_pair_key and (self.costs[i] + expertPairCost <= self.B): #Only add new experts that fit budget
                expert_skills = set(E_i)

                #Compute marginal coverage of new expert
                expert_coverage_total = len((expertPairSkills.union(expert_skills)).intersection(self.task_skills))/len(self.task)
                expert_marginal_cov = expert_coverage_total - expertPairCoverage
                expert_weight = expert_marginal_cov/self.costs[i]

                #push to maxheap - heapItem stored -gain, expert index and cost
                heapItem = (expert_weight*-1, i, self.costs[i])
                heappush(self.maxHeap2Guess, heapItem)

        return expertPairSkills, expertPairCoverage, expertPairCost
    
    
    def twoGuessPlainGreedy(self):
        '''
        2-Guess Plain Greedy from  Feldman, Nutov, Shoham 2021; Practical Budgeted Submodular Maximization
        '''
        startTime = time.perf_counter()

        allExpertPairs = {}
        #Get expert pairs and store union of skills and costs
        for i, expert_i in enumerate(self.experts):
            for j, expert_j in enumerate(self.experts):
                if i < j:
                    expert_pair_key = (i, j)
                    expert_pair_skills = set(expert_i).union(set(expert_j))
                    expert_pair_cost = self.costs[i] + self.costs[j]

                    #Only add experts who cost less than the budget
                    if expert_pair_cost <= self.B:
                        allExpertPairs[expert_pair_key] = [expert_pair_skills, expert_pair_cost]

        logging.info("Created allExpertPairs with {} pairs".format(len(allExpertPairs)))

        #Get best single expert solution
        best_single_expert, best_single_cov, best_single_cost = set(), 0, 0
        for i, expert_i in enumerate(self.experts):
            if self.costs[i] <= self.B:
                expert_i_cov = len(set(expert_i).intersection(self.task_skills))/len(self.task)
                if expert_i_cov > best_single_cov:
                    best_single_cov = expert_i_cov
                    best_single_cost = self.costs[i]
                    best_single_expert = set(expert_i)

        #Keep track of all solutions and their costs
        solutionDict = {}
        best_sol_experts, best_sol_skills, best_coverage, best_cost = [], set(), 0, 0

        #Run Plain Greedy for each pair
        for pair_key, pair_data in allExpertPairs.items():
            
            #Create priority queue with all other experts for this run
            #Initialize variables for this greedy run
            solution_skills, curr_coverage, curr_cost = self.createmaxHeap2Guess(expert_pair_key=pair_key, expert_pair_data=pair_data)
            solution_experts = [self.experts[pair_key[0]], self.experts[pair_key[1]]]

            #Assign experts greedily using maxHeap2Guess
            #Check if there is an element with cost that fits in budget
            while len(self.maxHeap2Guess) > 1 and (min(key[2] for key in self.maxHeap2Guess) <= (self.B - curr_cost)) and (curr_coverage < 1):
                
                #Pop best expert from maxHeap2Guess and compute marginal gain
                top_expert_key = heappop(self.maxHeap2Guess)
                top_expert_indx, top_expert_cost = top_expert_key[1], top_expert_key[2]
                top_expert_skills = set(self.experts[top_expert_indx]) #Get the skills of the top expert

                sol_with_top_expert = solution_skills.union(top_expert_skills)
                coverage_with_top_expert = len(sol_with_top_expert.intersection(self.task_skills))/len(self.task)
                top_expert_marginal_gain = (coverage_with_top_expert - curr_coverage)/top_expert_cost

                #Check expert now on top - 2nd expert on heap
                second_expert = self.maxHeap2Guess[0] 
                second_expert_heap_gain = second_expert[0]*-1

                #If marginal gain of top expert is better we add to solution
                if top_expert_marginal_gain >= second_expert_heap_gain:
                    #Only add if expert is within budget
                    if top_expert_cost + curr_cost <= self.B:
                        solution_skills = solution_skills.union(top_expert_skills)
                        solution_experts.append(self.experts[top_expert_indx])
                        curr_coverage = coverage_with_top_expert
                        curr_cost += top_expert_cost
                        logging.debug("Adding expert {}, curr_coverage={:.3f}, curr_cost={}".format(self.experts[top_expert_indx], curr_coverage, curr_cost))
                
                #Otherwise re-insert top expert into heap with updated marginal gain
                else:
                    updated_top_expert = (top_expert_marginal_gain*-1, top_expert_indx, top_expert_cost)
                    heappush(self.maxHeap2Guess, updated_top_expert)

            
            #Add solution to dict
            logging.info("Computed Pair Solution for seed{}, experts:{}, coverage={:.3f}, cost={}".format(pair_key, solution_experts, curr_coverage, curr_cost))
            solutionDict[pair_key] = {'experts':solution_experts, 'skills':solution_skills, 'coverage':curr_coverage, 'cost':curr_cost}
            if curr_coverage > best_coverage:
                best_coverage = curr_coverage
                best_cost = curr_cost
                best_sol_experts = solution_experts
                best_sol_skills = solution_skills

        #Compare with best single expert solution - if they are equivalent choose single
        if best_single_cov >= best_coverage:
            best_coverage = best_single_cov
            best_cost = best_single_cost
            best_sol_experts = list(best_single_expert)
            best_sol_skills = best_single_expert
        
        runTime = time.perf_counter() - startTime
        logging.info("2-Guess Plain Greedy Solution:{}, Coverage:{:.3f}, Cost:{}, Runtime = {:.2f} seconds".format(best_sol_experts, best_coverage, best_cost, runTime))

        return best_sol_experts, best_sol_skills, best_coverage, best_cost
    
    
    def oneGuessGreedyPlus(self):
        '''
        '''
        return

    

    def plotParetoCurve(self, coverageList, costList):
        '''
        Plot coverage (y-axis) vs. cost (x-axis) through one run of algorithm
        ARGS:
            coverageList : List of coverages
            costList     : List of costs
        '''
        plt.figure(figsize=(6, 4))
        plt.plot(costList, coverageList, '*', alpha=0.7)
        plt.title('Coverage vs. Cost')
        plt.ylabel("Task Coverage")
        plt.xlabel("Cost")
        plt.grid(alpha=0.3)
        plt.show()