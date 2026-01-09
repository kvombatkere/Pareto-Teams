import time, json, pickle
from heapq import heappop, heappush, heapify
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import logging
logging.basicConfig(format='%(asctime)s |%(levelname)s: %(message)s', level=logging.INFO)

class paretoKnapsackRestaurants():
    '''
    Define a class for restaurant recommendations with knapsack cost
    '''

    def __init__(self, n_items, costs, simMatrix, budget):
        '''
        Initialize instance with n items, similarity matrix and knapsack budget
        ARGS:
            n_items   : list of n items; each item is a list of skills
            costs       : cost of each item
            simMatrix   : similarity matrix between items
            budget      : knapsack budget
        '''
        self.items = n_items
        self.simMatrix = simMatrix
        self.n = len(self.items)
        self.costs = costs
        self.B = budget
        logging.info("Initialized Pareto Restaurant - Knapsack Cost Instance, Num Experts:{}, Budget={}".format(self.n, self.B))


    def computeSolutionObjective(self, solution_item_ids):
        '''
        Compute objective value of current solution
        ARGS:
            solution_item_ids : list of item indices in current solution
        RETURNS:
            objective_value   : objective value of current solution
        '''
        objective_value = 0

        #Compute c
        for i in range(self.n):
            max_sim = 0
            for item_id in solution_item_ids:
                if self.simMatrix[i][item_id] > max_sim:
                    max_sim = self.simMatrix[i][item_id]
            objective_value += max_sim

        return objective_value


    def getItemMarginalGain(self, item_index, current_solution):
        '''
        Compute marginal gain of adding item_index to current_solution
        ARGS:
            item_index       : index of item to compute marginal gain for
            current_solution : list of items in current solution
        RETURNS:
            marginal_gain   : marginal gain of adding item_index to current_solution
        '''
        marginal_gain = self.computeSolutionObjective(current_solution + [item_index]) - self.computeSolutionObjective(current_solution)
        
        return marginal_gain


    def createItemMaxHeap(self):
        '''
        Initialize self.maxHeap with item similarities for each item
        '''
        #Create max heap to store marginal gains computed from similarity matrix
        self.maxHeap = []
        heapify(self.maxHeap)
        
        for e in range(self.n):
            marginal_gain = 0
            for i in range(self.n):
                marginal_gain += self.simMatrix[i][e]

            #push to maxheap - heapItem stored -gain, item index and cost
            heapItem = (marginal_gain*-1, e, self.costs[e])
            heappush(self.maxHeap, heapItem)

        return 
    

    def plainGreedy(self):
        '''
        Adapt Plain Greedy Algorithm from  Feldman, Nutov, Shoham 2021; Practical Budgeted Submodular Maximization
        Run with input sets, self.experts instead of individual elements
        '''
        startTime = time.perf_counter()

        #Solution items and current objective and cost
        #Only track items by their indices
        curr_solution_items = [] 
        curr_objective, curr_cost = 0, 0

        #Create maxheap with coverages
        self.createItemMaxHeap()

        #Assign items greedily using max heap
        #Check if there is an element with cost that fits in budget
        while len(self.maxHeap) > 1 and (min(key[2] for key in self.maxHeap) <= (self.B - curr_cost)):
            
            #Pop best item from maxHeap and compute marginal gain
            top_item_key = heappop(self.maxHeap)
            top_item_indx, top_item_cost = top_item_key[1], top_item_key[2]

            objective_with_top_item = self.computeSolutionObjective(curr_solution_items + [top_item_indx])
            top_item_marginal_gain = (objective_with_top_item - self.computeSolutionObjective(curr_solution_items))/top_item_cost

            #Check expert now on top - 2nd expert on heap
            second_expert = self.maxHeap[0] 
            second_expert_heap_gain = second_expert[0]*-1

            #If marginal gain of top item is better we add to solution
            if top_item_marginal_gain >= second_expert_heap_gain:
                #Only add if item is within budget
                if top_item_cost + curr_cost <= self.B:
                    curr_solution_items.append(top_item_indx)
                    curr_objective = objective_with_top_item
                    curr_cost += top_item_cost
                    logging.info("Adding item {}, curr_coverage={:.3f}, curr_cost={}".format(self.items[top_item_indx], curr_objective, curr_cost))
            
            #Otherwise re-insert top item into heap with updated marginal gain
            else:
                updated_top_item = (top_item_marginal_gain*-1, top_item_indx, top_item_cost)
                heappush(self.maxHeap, updated_top_item)

        runTime = time.perf_counter() - startTime
        logging.info("Plain Greedy Solution:{}, Coverage:{:.3f}, Cost:{}, Runtime = {:.2f} seconds".format(curr_solution_items, curr_objective, curr_cost, runTime))

        return curr_solution_items, curr_objective, curr_cost, runTime
    

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

                feasible_expert_list, feasible_expert_skills, feasible_expert_cost = [], set(), 0
                #Perform Greedy+ check - Loop over solution in each iteration of plain greedy
                for i, expert_i in enumerate(solution_experts):
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
                            if added_expert_cov > seed_i_coverage:
                                seed_i_experts = feasible_expert_list.copy()
                                seed_i_experts.append(E_j)
                                seed_i_coverage = added_expert_cov
                                seed_i_cost = feasible_expert_cost + self.costs[j]
                                logging.debug("New feasible seed solution yielded better coverage! {}, coverage={:.3f}, cost={}".format(seed_i_experts,
                                                                                                                                       seed_i_coverage, seed_i_cost))
                                
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