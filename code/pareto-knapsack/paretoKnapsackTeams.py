import time, pickle
import random
from heapq import heappop, heappush, heapify
from typing import List, Tuple
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import logging
logging.basicConfig(format='%(asctime)s |%(levelname)s: %(message)s', level=logging.INFO)

class paretoKnapsackTeams():
    '''
    Define a class for coverage cost for n experts and single task, with knapsack cost
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
        self.costs = costs[:self.n]
        self.B = budget
        logging.debug("Initialized Pareto Coverage - Knapsack Cost Instance, Task:{}, Num Experts:{}, Budget={}".format(self.task, self.n, self.B))


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
                    logging.debug("Adding expert {}, curr_coverage={:.3f}, curr_cost={}".format(self.experts[top_expert_indx], curr_coverage, curr_cost))
            
            #Otherwise re-insert top expert into heap with updated marginal gain
            else:
                updated_top_expert = (top_expert_marginal_gain*-1, top_expert_indx, top_expert_cost)
                heappush(self.maxHeap, updated_top_expert)

        runTime = time.perf_counter() - startTime
        logging.debug("Plain Greedy Solution:{}, Coverage:{:.3f}, Cost:{}, Runtime = {:.2f} seconds".format(solution_experts, curr_coverage, curr_cost, runTime))

        return solution_experts, solution_skills, curr_coverage, curr_cost, runTime
    

    def greedyPlus(self):
        '''
        Greedy Plus Algorithm from  Feldman, Nutov, Shoham 2021; Practical Budgeted Submodular Maximization
        Greedy returns the better solution among the output of Plain Greedy and the best feasible solution 
        that can be obtained by combining any solution that Plain Greedy had at some iteration 
        with a single expert.
        '''
        startTime = time.perf_counter()

        #Get plain greedy solution
        sol_experts, sol_skills, best_coverage, best_cost, pg_runtime = self.plainGreedy()

        logging.debug("=="*50)
        best_experts_list, feasible_expert_list, feasible_expert_skills = [], [], set()
        feasible_expert_cost = 0

        #Loop over solution in each iteration of plain greedy
        for i, expert_i in enumerate(sol_experts):
            feasible_expert_list.append(expert_i)
            feasible_expert_skills = feasible_expert_skills.union(set(expert_i))
            feasible_expert_cost += self.costs[self.experts.index(expert_i)]
            logging.debug("Trying incremental solution:{}, cost:{}".format(feasible_expert_list, feasible_expert_cost))
            
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
                        logging.debug("New feasible solution yielded better coverage! {}, coverage={:.3f}, cost={}".format(best_experts_list,best_coverage,best_cost))
        
        #Return original solution if that is better
        if len(best_experts_list) == 0:
            logging.debug("Original Plain Greedy Solution was best!")
            best_experts_list = sol_experts

        runTime = time.perf_counter() - startTime
        logging.debug("Greedy+ Solution:{}, Coverage:{:.3f}, Cost:{}, Runtime = {:.2f} seconds".format(best_experts_list, best_coverage, best_cost, runTime))
        
        #Return solution
        return best_experts_list, sol_skills, best_coverage, best_cost, runTime


    def top_k(self):
        '''
        Budget-threshold heuristic: select experts by highest cost-scaled marginal gain
        with respect to the empty set (i.e., individual coverage / cost),
        adding experts until the budget is exhausted.
        Only considers experts that are individually within the budget.
        '''
        startTime = time.perf_counter()

        #Compute individual cost-scaled gains
        expert_scores = []
        for i, expert_i in enumerate(self.experts):
            if self.costs[i] <= self.B and self.costs[i] > 0:
                expert_cov = len(set(expert_i).intersection(self.task_skills)) / len(self.task)
                expert_scores.append((expert_cov / self.costs[i], i))

        #Sort by score descending and select until budget is exhausted
        expert_scores.sort(key=lambda x: x[0], reverse=True)
        selected_indices = []
        curr_cost = 0
        for _, idx in expert_scores:
            if curr_cost + self.costs[idx] <= self.B:
                selected_indices.append(idx)
                curr_cost += self.costs[idx]

        #Build solution
        solution_skills = set()
        solution_experts = []
        for idx in selected_indices:
            solution_experts.append(self.experts[idx])
            solution_skills = solution_skills.union(set(self.experts[idx]))

        curr_coverage = len(solution_skills.intersection(self.task_skills)) / len(self.task) if len(self.task) > 0 else 0
        runTime = time.perf_counter() - startTime
        logging.debug("Top-k (cost-scaled, budget-feasible) Solution, Coverage:{:.3f}, Cost:{}, Runtime = {:.2f} seconds".format(curr_coverage, curr_cost, runTime))

        return solution_experts, solution_skills, curr_coverage, curr_cost, runTime


    def random_heuristic(self):
        '''
        Random heuristic: randomly select experts until the budget is exhausted.
        Only considers experts that are individually within the budget.
        '''
        startTime = time.perf_counter()

        eligible_indices = [i for i, c in enumerate(self.costs) if 0 < c <= self.B]
        random.shuffle(eligible_indices)

        selected_indices = []
        curr_cost = 0
        for idx in eligible_indices:
            if curr_cost + self.costs[idx] <= self.B:
                selected_indices.append(idx)
                curr_cost += self.costs[idx]

        #Build solution
        solution_skills = set()
        solution_experts = []
        for idx in selected_indices:
            solution_experts.append(self.experts[idx])
            solution_skills = solution_skills.union(set(self.experts[idx]))

        curr_coverage = len(solution_skills.intersection(self.task_skills)) / len(self.task) if len(self.task) > 0 else 0
        runTime = time.perf_counter() - startTime
        logging.debug("Random (budget-feasible) Solution, Coverage:{:.3f}, Cost:{}, Runtime = {:.2f} seconds".format(curr_coverage, curr_cost, runTime))

        return solution_experts, solution_skills, curr_coverage, curr_cost, runTime
    

    def createmaxHeap2Guess(self, expert_pair_key, expert_pair_data):
        '''
        Initialize self.maxHeap2Guess with expert-task coverages for each expert that is not in the pair
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

        logging.debug("Created allExpertPairs with {} pairs".format(len(allExpertPairs)))

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
            logging.debug("Computed Pair Solution for seed{}, experts:{}, coverage={:.3f}, cost={}".format(pair_key, solution_experts, curr_coverage, curr_cost))
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
        logging.debug("2-Guess Plain Greedy Solution:{}, Coverage:{:.3f}, Cost:{}, Runtime = {:.2f} seconds".format(best_sol_experts, best_coverage, best_cost, runTime))

        return best_sol_experts, best_sol_skills, best_coverage, best_cost, runTime
    
    def prefixParetoGreedy_2Guess(self):
        '''
        Prefix Pareto Greedy Algorithm - implemented as a variant of 2-Guess Plain Greedy
        '''
        startTime = time.perf_counter()

        #Hashmap to track best coverage for each cost
        cost_coverage_map = {}
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

        logging.debug("Created allExpertPairs with {} pairs".format(len(allExpertPairs)))

        #Update single expert solutions
        for i, expert_i in enumerate(self.experts):
            if self.costs[i] <= self.B:
                expert_i_cov = len(set(expert_i).intersection(self.task_skills))/len(self.task)
                #Update cost coverage map
                if self.costs[i] not in cost_coverage_map or expert_i_cov > cost_coverage_map[self.costs[i]][0]:
                    cost_coverage_map[self.costs[i]] = [expert_i_cov, list(expert_i)]

        #Run Greedy for each pair and track prefixes
        for pair_key, pair_data in allExpertPairs.items():
            
            #Create priority queue with all other experts for this run
            #Initialize variables for this greedy run
            solution_skills, curr_coverage, curr_cost = self.createmaxHeap2Guess(expert_pair_key=pair_key, expert_pair_data=pair_data)
            solution_experts = [self.experts[pair_key[0]], self.experts[pair_key[1]]]
            
            #Update cost coverage map
            if curr_cost not in cost_coverage_map or curr_coverage > cost_coverage_map[curr_cost][0]:
                cost_coverage_map[curr_cost] = [curr_coverage, solution_experts.copy()]

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

                        #Update cost coverage map
                        if curr_cost not in cost_coverage_map or curr_coverage > cost_coverage_map[curr_cost][0]:
                            cost_coverage_map[curr_cost] = [curr_coverage, solution_experts.copy()]
                        logging.debug("Adding expert {}, curr_coverage={:.3f}, curr_cost={}".format(self.experts[top_expert_indx], curr_coverage, curr_cost))
                
                #Otherwise re-insert top expert into heap with updated marginal gain
                else:
                    updated_top_expert = (top_expert_marginal_gain*-1, top_expert_indx, top_expert_cost)
                    heappush(self.maxHeap2Guess, updated_top_expert)

        #Prune cost_coverage_map to only keep Pareto optimal solutions
        prunedBudgets, prunedCoverages = [], []
        currentCov = 0
        for b_prime in sorted(cost_coverage_map.keys()):
            if cost_coverage_map[b_prime][0] > currentCov:
                currentCov = cost_coverage_map[b_prime][0]
                prunedBudgets.append(b_prime)
                prunedCoverages.append(currentCov)
                logging.debug("Approx. Pareto Budget: {}, Coverage: {}, Experts: {}".format(b_prime, cost_coverage_map[b_prime][0], cost_coverage_map[b_prime][1]))

        runTime = time.perf_counter() - startTime
        logging.debug("Prefix Pareto Greedy Runtime = {:.2f} seconds".format(runTime))

        return prunedBudgets, prunedCoverages, cost_coverage_map, runTime
    

    def createmaxHeap1Guess(self, seed_expert, seed_expert_cost, seed_expert_index):
        '''
        Initialize self.maxHeap1Guess with expert-task coverages for each expert that is not the seed
        '''
        #Create max heap to store coverages with respect to new objective function
        self.maxHeap1Guess = []
        heapify(self.maxHeap1Guess)

        #Compute skills, cost and coverage of pair
        expertCoverage = len(set(seed_expert).intersection(self.task_skills))/len(self.task)
        
        for i, E_i in enumerate(self.experts):
            if i != seed_expert_index and (self.costs[i] + seed_expert_cost <= self.B): #Only add new experts that fit budget
                expert_skills = set(E_i)

                #Compute marginal coverage of new expert
                expert_coverage_total = len((set(seed_expert).union(expert_skills)).intersection(self.task_skills))/len(self.task)
                expert_marginal_cov = expert_coverage_total - expertCoverage
                expert_weight = expert_marginal_cov/self.costs[i]

                #push to maxheap - heapItem stored -gain, expert index and cost
                heapItem = (expert_weight*-1, i, self.costs[i])
                heappush(self.maxHeap1Guess, heapItem)

        return expertCoverage, seed_expert_cost
        

    def oneGuessGreedyPlus(self):
        '''
        1-Guess Greedy+ from Feldman, Nutov, Shoham 2021; Practical Budgeted Submodular Maximization
        '''
        startTime = time.perf_counter()

        #Keep track of all solutions and their costs
        solutionDict = {}
        best_sol_experts, best_sol_skills, best_coverage, best_cost = [], set(), 0, 0

        #Iterate over all single expert seeds
        for i, expert_i in enumerate(self.experts):
            if self.costs[i] <= self.B:
                expert_i_cov = len(set(expert_i).intersection(self.task_skills))/len(self.task) 

                #Create priority queue with all other experts for this run
                #Initialize variables for this greedy run
                curr_coverage, curr_cost = self.createmaxHeap1Guess(seed_expert=expert_i, 
                                                                    seed_expert_cost=self.costs[i], 
                                                                    seed_expert_index=i)
                
                solution_skills, solution_experts = set(expert_i), [expert_i]

                #Assign experts greedily using max heap
                #Check if there is an element with cost that fits in budget
                while len(self.maxHeap1Guess) > 1 and (min(key[2] for key in self.maxHeap1Guess) <= (self.B - curr_cost)) and (curr_coverage < 1):
                    
                    #Pop best expert from maxHeap1Guess and compute marginal gain
                    top_expert_key = heappop(self.maxHeap1Guess)
                    top_expert_indx, top_expert_cost = top_expert_key[1], top_expert_key[2]
                    top_expert_skills = set(self.experts[top_expert_indx]) #Get the skills of the top expert

                    sol_with_top_expert = solution_skills.union(top_expert_skills)
                    coverage_with_top_expert = len(sol_with_top_expert.intersection(self.task_skills))/len(self.task)
                    top_expert_marginal_gain = (coverage_with_top_expert - curr_coverage)/top_expert_cost

                    #Check expert now on top - 2nd expert on heap
                    second_expert = self.maxHeap1Guess[0] 
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
                        heappush(self.maxHeap1Guess, updated_top_expert)

                #Store results for run with seed i
                seed_i_coverage, seed_i_cost = curr_coverage, curr_cost
                seed_i_experts, seed_i_skills = solution_experts.copy(), solution_skills

                # feasible_expert_list, feasible_expert_skills, feasible_expert_cost = [], set(), 0
                # #Perform Greedy+ check - Loop over solution in each iteration of plain greedy
                # for i, expert_i in enumerate(solution_experts):
                #     feasible_expert_list.append(expert_i)
                #     feasible_expert_skills = feasible_expert_skills.union(set(expert_i))
                #     feasible_expert_cost += self.costs[self.experts.index(expert_i)]
                #     logging.debug("Trying incremental solution:{}, cost:{}".format(feasible_expert_list, feasible_expert_cost))
                    
                #     for j, E_j in enumerate(self.experts):
                #         #If adding a single expert doesn't violate budget
                #         if feasible_expert_cost + self.costs[j] <= self.B:
                #             #Compute coverage by adding expert to incremental solution
                #             added_expert_cov = len((feasible_expert_skills.union(set(E_j))).intersection(self.task_skills))/len(self.task)
                            
                #             #If this solution is better than original solution, store it
                #             if added_expert_cov > seed_i_coverage:
                #                 seed_i_experts = feasible_expert_list.copy()
                #                 seed_i_experts.append(E_j)
                #                 seed_i_coverage = added_expert_cov
                #                 seed_i_cost = feasible_expert_cost + self.costs[j]
                #                 logging.debug("New feasible seed solution yielded better coverage! {}, coverage={:.3f}, cost={}".format(seed_i_experts,
                #                                                                                               seed_i_coverage, seed_i_cost))
                                
                #Store best solution for seed i
                logging.debug("Best solution for seed {}, experts:{}, coverage={:.3f}, cost={}".format(i, seed_i_experts, seed_i_coverage, seed_i_cost))
                solutionDict[i] = {'experts':seed_i_experts, 'skills':seed_i_skills, 'coverage':seed_i_coverage, 'cost':seed_i_cost}
                #Keep track of best solution across all seeds
                if seed_i_coverage > best_coverage:
                    best_coverage = seed_i_coverage
                    best_cost = seed_i_cost
                    best_sol_experts = seed_i_experts
                    best_sol_skills = seed_i_skills

        runTime = time.perf_counter() - startTime
        logging.debug("1-Guess Greedy+ Solution:{}, Coverage:{:.3f}, Cost:{}, Runtime = {:.2f} seconds".format(best_sol_experts, best_coverage, best_cost, runTime))

        return best_sol_experts, best_sol_skills, best_coverage, best_cost, runTime


    def prefixParetoGreedy_1Guess(self):
        '''
        Prefix Pareto Greedy Algorithm - implemented as a variant of 1-Guess Greedy
        '''
        startTime = time.perf_counter()

        #Hashmap to track best coverage for each cost
        cost_coverage_map = {}

        #Iterate over all single expert seeds
        for i, expert_i in enumerate(self.experts):
            if self.costs[i] <= self.B:
                expert_i_cov = len(set(expert_i).intersection(self.task_skills))/len(self.task) 

                #Update cost coverage map
                if self.costs[i] not in cost_coverage_map or expert_i_cov > cost_coverage_map[self.costs[i]][0]:
                    cost_coverage_map[self.costs[i]] = [expert_i_cov, list(expert_i)]

                #Create priority queue with all other experts for this run
                #Initialize variables for this greedy run
                curr_coverage, curr_cost = self.createmaxHeap1Guess(seed_expert=expert_i, seed_expert_cost=self.costs[i], 
                                                                    seed_expert_index=i)
                solution_skills, solution_experts = set(expert_i), [expert_i]

                #Assign experts greedily using max heap
                #Check if there is an element with cost that fits in budget
                while len(self.maxHeap1Guess) > 1 and (min(key[2] for key in self.maxHeap1Guess) <= (self.B - curr_cost)) and (curr_coverage < 1):
                    
                    #Pop best expert from maxHeap1Guess and compute marginal gain
                    top_expert_key = heappop(self.maxHeap1Guess)
                    top_expert_indx, top_expert_cost = top_expert_key[1], top_expert_key[2]
                    top_expert_skills = set(self.experts[top_expert_indx]) #Get the skills of the top expert

                    sol_with_top_expert = solution_skills.union(top_expert_skills)
                    coverage_with_top_expert = len(sol_with_top_expert.intersection(self.task_skills))/len(self.task)
                    top_expert_marginal_gain = (coverage_with_top_expert - curr_coverage)/top_expert_cost

                    #Check expert now on top - 2nd expert on heap
                    second_expert = self.maxHeap1Guess[0] 
                    second_expert_heap_gain = second_expert[0]*-1

                    #If marginal gain of top expert is better we add to solution
                    if top_expert_marginal_gain >= second_expert_heap_gain:
                        #Only add if expert is within budget
                        if top_expert_cost + curr_cost <= self.B:
                            solution_skills = solution_skills.union(top_expert_skills)
                            solution_experts.append(self.experts[top_expert_indx])
                            curr_coverage = coverage_with_top_expert
                            curr_cost += top_expert_cost

                            #Update cost coverage map
                            if curr_cost not in cost_coverage_map or curr_coverage > cost_coverage_map[curr_cost][0]:
                                cost_coverage_map[curr_cost] = [curr_coverage, solution_experts.copy()]
                            logging.debug("Adding expert {}, curr_coverage={:.3f}, curr_cost={}".format(self.experts[top_expert_indx], curr_coverage, curr_cost))
                    
                    #Otherwise re-insert top expert into heap with updated marginal gain
                    else:
                        updated_top_expert = (top_expert_marginal_gain*-1, top_expert_indx, top_expert_cost)
                        heappush(self.maxHeap1Guess, updated_top_expert)

        #Prune cost_coverage_map to only keep Pareto optimal solutions
        prunedBudgets, prunedCoverages = [], []
        currentCov = 0
        for b_prime in sorted(cost_coverage_map.keys()):
            if cost_coverage_map[b_prime][0] > currentCov:
                currentCov = cost_coverage_map[b_prime][0]
                prunedBudgets.append(b_prime)
                prunedCoverages.append(currentCov)
                logging.debug("Approx. Pareto Budget: {}, Coverage: {}, Experts: {}".format(b_prime, cost_coverage_map[b_prime][0], cost_coverage_map[b_prime][1]))

        runTime = time.perf_counter() - startTime
        logging.debug("Prefix Pareto Greedy - 1 Guess Runtime = {:.2f} seconds".format(runTime))

        return prunedBudgets, prunedCoverages, cost_coverage_map, runTime
    

    def F_Greedy(self):
        '''
        Linear coverage sweep: for each discrete coverage level, find a minimum-cost
        solution using weighted greedy (marginal gain scaled by cost) with seed size 1.
        Then prune dominated solutions.
        '''
        startTime = time.perf_counter()

        # Discrete coverage targets: 1/|task|, 2/|task|, ..., 1
        if len(self.task) == 0:
            return [], [], {}, 0.0

        target_coverages = [k / len(self.task) for k in range(1, len(self.task) + 1)]

        # Track best solution per target coverage
        cost_coverage_map = {}

        for cov_x in target_coverages:
            # Try all single-expert seeds for this target coverage
            for i, expert_i in enumerate(self.experts):
                if self.costs[i] > self.B:
                    continue

                curr_coverage, curr_cost = self.createmaxHeap1Guess(
                    seed_expert=expert_i,
                    seed_expert_cost=self.costs[i],
                    seed_expert_index=i
                )
                solution_skills = set(expert_i)
                solution_experts = [expert_i]

                # Weighted greedy until reaching target coverage or no feasible expert
                while len(self.maxHeap1Guess) > 1 and (min(key[2] for key in self.maxHeap1Guess) <= (self.B - curr_cost)) and (curr_coverage < cov_x):
                    top_expert_key = heappop(self.maxHeap1Guess)
                    top_expert_indx, top_expert_cost = top_expert_key[1], top_expert_key[2]
                    top_expert_skills = set(self.experts[top_expert_indx])

                    sol_with_top_expert = solution_skills.union(top_expert_skills)
                    coverage_with_top_expert = len(sol_with_top_expert.intersection(self.task_skills)) / len(self.task)
                    top_expert_marginal_gain = (coverage_with_top_expert - curr_coverage) / top_expert_cost

                    # Compare against next best heap gain
                    second_expert = self.maxHeap1Guess[0]
                    second_expert_heap_gain = second_expert[0] * -1

                    if top_expert_marginal_gain >= second_expert_heap_gain:
                        if top_expert_cost + curr_cost <= self.B:
                            solution_skills = solution_skills.union(top_expert_skills)
                            solution_experts.append(self.experts[top_expert_indx])
                            curr_coverage = coverage_with_top_expert
                            curr_cost += top_expert_cost
                            logging.debug("Adding expert {}, curr_coverage={:.3f}, curr_cost={}".format(self.experts[top_expert_indx], curr_coverage, curr_cost))
                    else:
                        updated_top_expert = (top_expert_marginal_gain * -1, top_expert_indx, top_expert_cost)
                        heappush(self.maxHeap1Guess, updated_top_expert)

                # Store if target met within budget
                if curr_coverage >= cov_x:
                    # Keep minimum cost for this coverage
                    if cov_x not in cost_coverage_map or curr_cost < cost_coverage_map[cov_x][0]:
                        cost_coverage_map[cov_x] = [curr_cost, solution_experts.copy()]

        # Prune dominated solutions: keep strictly increasing coverage as cost increases
        prunedBudgets, prunedCoverages = [], []
        pairs = [(data[0], cov) for cov, data in cost_coverage_map.items()]
        pairs.sort(key=lambda x: x[0])  # sort by cost
        best_cov = -1.0
        for cost, cov in pairs:
            if cov > best_cov:
                best_cov = cov
                prunedBudgets.append(cost)
                prunedCoverages.append(cov)
                logging.debug("Approx. Pareto Coverage: {}, Cost: {}, Experts: {}".format(cov, cost, cost_coverage_map[cov][1]))

        runTime = time.perf_counter() - startTime
        logging.debug("Coverage Linear Runtime = {:.2f} seconds".format(runTime))

        return prunedBudgets, prunedCoverages, cost_coverage_map, runTime

    
    
    def ParetoPoint(self, budget):
        '''
        Compute a Pareto optimal point for the given budget.
        Uses 1-Guess Greedy Plus to find the best expert team within budget.
        
        ARGS:
            budget: The budget constraint
            
        RETURN:
            experts: List of selected experts
            coverage: Coverage value achieved
        '''
        # Use 1-Guess Greedy Plus to find best solution within budget
        best_experts, best_skills, best_coverage, best_cost, _ = self.oneGuessGreedyPlus()
        
        # If the solution fits within budget, return it
        if best_cost <= budget:
            return best_experts, best_coverage
        
        # Otherwise, greedily select experts within budget
        solution_experts = []
        solution_skills = set()
        curr_cost = 0
        
        expert_scores = []
        for i, expert_i in enumerate(self.experts):
            if self.costs[i] > 0:
                expert_cov = len(set(expert_i).intersection(self.task_skills)) / len(self.task)
                expert_scores.append((expert_cov / self.costs[i], i))
        
        expert_scores.sort(key=lambda x: x[0], reverse=True)
        
        for _, idx in expert_scores:
            if curr_cost + self.costs[idx] <= budget:
                solution_experts.append(self.experts[idx])
                solution_skills = solution_skills.union(set(self.experts[idx]))
                curr_cost += self.costs[idx]
        
        coverage = len(solution_skills.intersection(self.task_skills)) / len(self.task) if len(self.task) > 0 else 0
        return solution_experts, coverage


    def Pass(self, l, r, delta_val):
        '''
        Test if the interval [l, r] passes the certification criterion.
        Checks if the chord between (l, v_l) and (r, v_r) approximately covers 
        the value at the midpoint.
        
        ARGS:
            l: Left endpoint budget
            r: Right endpoint budget
            delta_val: Tolerance for certification
            
        RETURN:
            passes: True if interval passes the test, False otherwise
        '''
        # Compute Pareto points at endpoints
        S_l, v_l = self.ParetoPoint(l)
        S_r, v_r = self.ParetoPoint(r)
        
        # Compute midpoint (arithmetic midpoint)
        B_m = (l + r) / 2.0
        S_m, v_m = self.ParetoPoint(B_m)
        
        # Compute chord value at midpoint
        v_L_m = ((r - B_m) / (r - l)) * v_l + ((B_m - l) / (r - l)) * v_r
        
        # Return True if chord value is at least (1-delta) times actual value
        passes = v_L_m >= (1 - delta_val) * v_m
        logging.debug("Pass(l={}, r={}): v_L_m={:.4f}, (1-delta)*v_m={:.4f}, result={}".format(
            l, r, v_L_m, (1 - delta_val) * v_m, passes))
        return passes


    def exponentialSearchRepresentativeIntervals(self, B_min, B_max, epsilon_val=0.1, delta_val=0.1):
        '''
        Exponential Search for Representative Pareto Intervals
        
        Finds a set of representative intervals that approximate the Pareto frontier
        using exponential search and binary search.
        
        Algorithm:
        - Maintains a left endpoint l, initially B_min
        - For each iteration: exponentially searches for the largest r such that 
          Pass(l, r) is True using steps of r(1+epsilon)^2
        - If last exponential step passes, includes it; otherwise uses binary search 
          between the last passing r_prev and failing r
        - Records the Pareto point at l and interval [l, r']
        - Advances l to r' and repeats until l >= B_max
        
        ARGS:
            B_min: Minimum budget
            B_max: Maximum budget
            epsilon_val: Grid parameter, epsilon > 0 (default 0.1)
            delta_val: Tolerance, delta in (0,1) (default 0.1)
            
        RETURN:
            P_R: List of (experts, interval) tuples representing the Pareto frontier
            metadata: Dictionary with algorithm details
        '''
        startTime = time.perf_counter()
        
        P_R = []  # Representative Pareto intervals
        l = B_min
        
        def BinarySearchLargestPassing(l_left, l_right, r_right):
            '''
            Binary search for largest budget r' in [l_right, r_right] such that 
            Pass(l_left, r') returns True.
            
            ARGS:
                l_left: Left endpoint (fixed during binary search)
                l_right: Lower bound for binary search
                r_right: Upper bound for binary search
                
            RETURN:
                r_prime: Largest budget in [l_right, r_right] where Pass(l_left, r') is True
            '''
            lo = l_right
            hi = r_right
            
            # Floating-point binary search
            while hi - lo > 1e-6:  # Tolerance for convergence
                mid = (lo + hi) / 2.0
                if self.Pass(l_left, mid, delta_val):
                    lo = mid
                else:
                    hi = mid
            
            return lo
        
        iteration = 0
        while l < B_max:
            iteration += 1
            logging.info("Exponential Search Iteration {}: l={}".format(iteration, l))
            
            # Initialize r = l(1+epsilon)^2
            r = min(l * ((1 + epsilon_val) ** 2), B_max)
            r_prev = l
            
            # Exponential search phase: keep doubling r while Pass(l, r) is True
            while self.Pass(l, r, delta_val):
                logging.debug("Exponential search: Pass(l={}, r={}) = True, advancing r".format(l, r))
                r_prev = r
                r_uncapped = r * ((1 + epsilon_val) ** 2)
                
                # If next step would exceed B_max, stop expanding
                if r_uncapped >= B_max:
                    r = B_max  # Mark that we've hit the budget limit
                    break
                r = r_uncapped
            
            # Determine final r'
            if r >= B_max:
                # Exponential search reached budget limit, don't expand further
                r_prime = r_prev
                logging.debug("Exponential search reached B_max, using r_prev={}".format(r_prev))
            elif self.Pass(l, r, delta_val):
                # Last exponential step passed
                r_prime = r
                logging.debug("Pass(l={}, r={}) = True, setting r'={}".format(l, r, r_prime))
            else:
                # Last exponential step failed: binary search between r_prev and r
                r_prime = BinarySearchLargestPassing(l, r_prev, r)
                logging.debug("Pass(l={}, r={}) = False, binary search between {} and {} returned r'={}".format(
                    l, r, r_prev, r, r_prime))
            
            # Get Pareto point at l
            R_l, v_l = self.ParetoPoint(l)
            
            # Check if full coverage achieved at left endpoint
            if v_l >= 1.0 - 1e-6:
                # Add final interval and stop
                P_R.append((R_l, (l, r_prime)))
                logging.info("Added interval: [l={:.2f}, r'={:.2f}], v_l={:.3f}, experts={}".format(
                    l, r_prime, v_l, len(R_l)))
                logging.info("Full coverage (v_l={:.4f}) achieved at l={:.2f}, stopping search".format(v_l, l))
                break
            
            # Add to representative intervals
            P_R.append((R_l, (l, r_prime)))
            logging.info("Added interval: [l={:.2f}, r'={:.2f}], v_l={:.3f}, experts={}".format(
                l, r_prime, v_l, len(R_l)))
            
            # Update l for next iteration
            l = r_prime
        
        runTime = time.perf_counter() - startTime
        
        metadata = {
            'epsilon': epsilon_val,
            'delta': delta_val,
            'B_min': B_min,
            'B_max': B_max,
            'num_intervals': len(P_R),
            'runtime': runTime
        }
        
        logging.info("Exponential Search Complete: {} intervals found, Runtime={:.2f}s".format(
            len(P_R), runTime))
        
        return P_R, metadata
    

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