import time, json, pickle
from heapq import heappop, heappush, heapify
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
            n_items     : list of n items; each item is a restaurant
            costs       : cost of each item
            simMatrix   : similarity matrix between items
            budget      : knapsack budget
        '''
        self.items = n_items
        self.simMatrix = simMatrix
        self.n = len(self.items)
        self.costs = costs
        self.B = budget
        logging.info("Initialized Pareto Restaurant - Knapsack Cost Instance, Num Items:{}, Budget={}".format(self.n, self.B))


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
            current_solution : list of indices of items in current solution
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
        '''
        startTime = time.perf_counter()

        #Solution items and current objective and cost
        #Only track items by their indices
        curr_solution_items = [] 
        curr_objective, curr_cost = 0, 0

        #Create maxheap with objectives
        self.createItemMaxHeap()

        #Assign items greedily using max heap
        #Check if there is an element with cost that fits in budget
        while len(self.maxHeap) > 1 and (min(key[2] for key in self.maxHeap) <= (self.B - curr_cost)):
            
            #Pop best item from maxHeap and compute marginal gain
            top_item_key = heappop(self.maxHeap)
            top_item_indx, top_item_cost = top_item_key[1], top_item_key[2]

            objective_with_top_item = self.computeSolutionObjective(curr_solution_items + [top_item_indx])
            top_item_marginal_gain = (objective_with_top_item - self.computeSolutionObjective(curr_solution_items))/top_item_cost

            #Check item now on top - 2nd item on heap
            second_itembest_single_item = self.maxHeap[0] 
            second_item_heap_gain = second_itembest_single_item[0]*-1

            #If marginal gain of top item is better we add to solution
            if top_item_marginal_gain >= second_item_heap_gain:
                #Only add if item is within budget
                if top_item_cost + curr_cost <= self.B:
                    curr_solution_items.append(top_item_indx)
                    curr_objective = objective_with_top_item
                    curr_cost += top_item_cost
                    logging.debug("Adding item {}, curr_objective={:.3f}, curr_cost={}".format(self.items[top_item_indx], curr_objective, curr_cost))
            
            #Otherwise re-insert top item into heap with updated marginal gain
            else:
                updated_top_item = (top_item_marginal_gain*-1, top_item_indx, top_item_cost)
                heappush(self.maxHeap, updated_top_item)

        runTime = time.perf_counter() - startTime
        logging.debug("Plain Greedy Solution:{}, Objective:{:.3f}, Cost:{}, Runtime = {:.2f} seconds".format(curr_solution_items, curr_objective, curr_cost, runTime))

        return curr_solution_items, curr_objective, curr_cost, runTime
    

    def greedyPlus(self):
        '''
        Greedy Plus Algorithm from  Feldman, Nutov, Shoham 2021; Practical Budgeted Submodular Maximization
        Greedy returns the better solution among the output of Plain Greedy and the best feasible solution 
        that can be obtained by combining any solution that Plain Greedy had at some iteration 
        with a single item.
        '''
        startTime = time.perf_counter()

        #Get plain greedy solution
        sol_items, best_objective, best_cost, pg_runtime = self.plainGreedy()

        logging.debug("=="*50)
        best_items_list, feasible_item_list = [], []
        feasible_item_cost = 0

        #Loop over solution in each iteration of plain greedy
        for i, item_i in enumerate(sol_items):
            feasible_item_list.append(item_i)
            feasible_item_cost += self.costs[item_i]
            logging.debug("Trying incremental solution:{}, cost:{}".format(feasible_item_list, feasible_item_cost))
            
            for j, item_j in enumerate(self.items):
                #If adding a single item doesn't violate budget
                if feasible_item_cost + self.costs[j] <= self.B:
                    #Compute objective by adding item to incremental solution
                    added_item_obj = self.computeSolutionObjective(feasible_item_list + [j])
                    
                    #If this solution is better than original solution, store it
                    if added_item_obj > best_objective:
                        best_items_list = feasible_item_list.copy()
                        best_items_list.append(j)
                        best_objective = added_item_obj
                        best_cost = feasible_item_cost + self.costs[j]
                        logging.debug("New feasible solution yielded better objective! {}, objective={:.3f}, cost={}".format(best_items_list,best_objective,best_cost))
        
        #Return original solution if that is better
        if len(best_items_list) == 0:
            logging.debug("Original Plain Greedy Solution was best!")
            best_items_list = sol_items

        runTime = time.perf_counter() - startTime
        logging.debug("Greedy+ Solution:{}, Objective:{:.3f}, Cost:{}, Runtime = {:.2f} seconds".format(best_items_list, best_objective,best_cost,runTime))
        
        #Return solution
        return best_items_list, best_objective, best_cost, runTime


    def top_k(self):
        '''
        Budget-threshold heuristic: select items by highest cost-scaled marginal gain
        with respect to the empty set (i.e., sum of similarities / cost),
        adding items until the budget is exhausted.
        Only considers items that are individually within the budget.
        '''
        startTime = time.perf_counter()

        item_scores = []
        for e in range(self.n):
            cost = self.costs[e]
            if cost <= self.B and cost > 0:
                marginal_gain = float(np.sum(self.simMatrix[:, e]))
                item_scores.append((marginal_gain / cost, e))

        item_scores.sort(key=lambda x: x[0], reverse=True)
        selected_indices = []
        curr_cost = 0.0
        for _, idx in item_scores:
            if curr_cost + self.costs[idx] <= self.B:
                selected_indices.append(idx)
                curr_cost += self.costs[idx]

        curr_objective = self.computeSolutionObjective(selected_indices) if selected_indices else 0

        runTime = time.perf_counter() - startTime
        logging.debug("Top-k (cost-scaled, budget-feasible) Solution, Objective:{:.3f}, Cost:{}, Runtime = {:.2f} seconds".format(curr_objective, curr_cost, runTime))

        return selected_indices, curr_objective, curr_cost, runTime
    

    def createmaxHeap2Guess(self, item_pair_key, item_pair_cost):
        '''
        Initialize self.maxHeap2Guess with item-task objectives for each item that is not in the pair
        '''
        #Create max heap to store objectives with respect to new objective function
        self.maxHeap2Guess = []
        heapify(self.maxHeap2Guess)

        #Compute cost and objective of pair
        itemPairSol = [item_pair_key[0], item_pair_key[1]]
        itemPairCost = item_pair_cost
        itemPairobjective = self.computeSolutionObjective(itemPairSol)
        
        for i, item_i in enumerate(self.items):
            if i not in item_pair_key and (self.costs[i] + itemPairCost <= self.B): #Only add new items that fit budget
                #Compute marginal objective of new item
                item_marginal_gain = self.computeSolutionObjective(itemPairSol + [i]) - itemPairobjective
                item_weight = item_marginal_gain/self.costs[i]

                #push to maxheap - heapItem stored -gain, item index and cost
                heapItem = (item_weight*-1, i, self.costs[i])
                heappush(self.maxHeap2Guess, heapItem)

        return itemPairobjective, itemPairCost, itemPairSol
    

    def twoGuessPlainGreedy(self):
        '''
        2-Guess Plain Greedy from  Feldman, Nutov, Shoham 2021; Practical Budgeted Submodular Maximization
        '''
        startTime = time.perf_counter()

        allItemPairs = {}
        #Get item pairs and store union of skills and costs
        for i, item_i in enumerate(self.items):
            for j, item_j in enumerate(self.items):
                if i < j:
                    item_pair_key = (i, j)
                    item_pair_cost = self.costs[i] + self.costs[j]

                    #Only add items who cost less than the budget
                    if item_pair_cost <= self.B:
                        allItemPairs[item_pair_key] = item_pair_cost

        logging.debug("Created allItemPairs with {} pairs".format(len(allItemPairs)))

        #Get best single item solution
        best_single_item, best_single_obj, best_single_cost = set(), 0, 0
        for i, item_i in enumerate(self.items):
            if self.costs[i] <= self.B:
                item_i_obj = self.computeSolutionObjective([i])

                if item_i_obj > best_single_obj:
                    best_single_obj = item_i_obj
                    best_single_cost = self.costs[i]
                    best_single_item = item_i

        #Keep track of all solutions and their costs
        solutionDict = {}
        best_sol_items, best_objective, best_cost = [], 0, 0

        #Run Plain Greedy for each pair
        for pair_key, pair_cost in allItemPairs.items():
            
            #Create priority queue with all other items for this run
            #Initialize variables for this greedy run
            curr_objective, curr_cost, curr_solution_items = self.createmaxHeap2Guess(item_pair_key=pair_key, item_pair_cost=pair_cost)

            #Assign items greedily using maxHeap2Guess
            #Check if there is an element with cost that fits in budget
            while len(self.maxHeap2Guess) > 0 and (min(key[2] for key in self.maxHeap2Guess) <= (self.B - curr_cost)):
                
                #Pop best item from maxHeap and compute marginal gain
                top_item_key = heappop(self.maxHeap2Guess)
                top_item_indx, top_item_cost = top_item_key[1], top_item_key[2]

                objective_with_top_item = self.computeSolutionObjective(curr_solution_items + [top_item_indx])
                top_item_marginal_gain = (objective_with_top_item - self.computeSolutionObjective(curr_solution_items))/top_item_cost

                #Check item now on top - 2nd item on heap
                if len(self.maxHeap2Guess) > 0:
                    second_item = self.maxHeap2Guess[0]
                    second_item_heap_gain = second_item[0]*-1
                else:
                    second_item_heap_gain = float("-inf")

                #If marginal gain of top item is better we add to solution
                if top_item_marginal_gain >= second_item_heap_gain:
                    #Only add if item is within budget
                    if top_item_cost + curr_cost <= self.B:
                        curr_solution_items.append(top_item_indx)
                        curr_objective = objective_with_top_item
                        curr_cost += top_item_cost
                        logging.debug("Adding item {}, curr_objective={:.3f}, curr_cost={}".format(self.items[top_item_indx], curr_objective, curr_cost))
            
                #Otherwise re-insert top item into heap with updated marginal gain
                else:
                    updated_top_item = (top_item_marginal_gain*-1, top_item_indx, top_item_cost)
                    heappush(self.maxHeap2Guess, updated_top_item)

            #Add solution to dict
            logging.debug("Computed Pair Solution for seed{}, items:{}, objective={:.3f}, cost={}".format(pair_key, curr_solution_items, curr_objective, curr_cost))
            solutionDict[pair_key] = {'items':curr_solution_items, 'objective':curr_objective, 'cost':curr_cost}
            if curr_objective > best_objective:
                best_objective = curr_objective
                best_cost = curr_cost
                best_sol_items = curr_solution_items

        #Compare with best single item solution - if they are equivalent choose single
        if best_single_obj >= best_objective:
            best_objective = best_single_obj
            best_cost = best_single_cost
            best_sol_items = list(best_single_item)
        
        runTime = time.perf_counter() - startTime
        logging.debug("2-Guess Plain Greedy Solution:{}, objective:{:.3f}, Cost:{}, Runtime = {:.2f} seconds".format(best_sol_items, best_objective, best_cost, runTime))

        return best_sol_items, best_objective, best_cost, runTime
    

    def prefixParetoGreedy_2Guess(self):
        '''
        Prefix Pareto Greedy Algorithm - implemented as a variant of 2-Guess Plain Greedy
        '''
        startTime = time.perf_counter()

        #Hashmap to track best objective for each cost
        cost_objective_map = {}
        allItemPairs = {}

        #Get item pairs and store union of skills and costs
        for i, item_i in enumerate(self.items):
            for j, item_j in enumerate(self.items):
                if i < j:
                    item_pair_key = (i, j)
                    item_pair_cost = self.costs[i] + self.costs[j]

                    #Only add items who cost less than the budget
                    if item_pair_cost <= self.B:
                        allItemPairs[item_pair_key] = item_pair_cost

        logging.debug("Created allItemPairs with {} pairs".format(len(allItemPairs)))

        #Update single item solutions
        for i, item_i in enumerate(self.items):
            if self.costs[i] <= self.B:
                item_i_obj = self.computeSolutionObjective([i])
                #Update cost objective map
                if self.costs[i] not in cost_objective_map or item_i_obj > cost_objective_map[self.costs[i]][0]:
                    cost_objective_map[self.costs[i]] = [item_i_obj, [i]]

        #Run Greedy for each pair and track prefixes
        for pair_key, pair_cost in allItemPairs.items():
            
            #Create priority queue with all other items for this run
            #Initialize variables for this greedy run
            curr_objective, curr_cost, curr_solution_items = self.createmaxHeap2Guess(item_pair_key=pair_key, item_pair_cost=pair_cost)
            
            #Update cost objective map
            if curr_cost not in cost_objective_map or curr_objective > cost_objective_map[curr_cost][0]:
                cost_objective_map[curr_cost] = [curr_objective, curr_solution_items.copy()]

            #Assign items greedily using maxHeap2Guess
            #Check if there is an element with cost that fits in budget
            while len(self.maxHeap2Guess) > 1 and (min(key[2] for key in self.maxHeap2Guess) <= (self.B - curr_cost)):
                
                 #Pop best item from maxHeap and compute marginal gain
                top_item_key = heappop(self.maxHeap2Guess)
                top_item_indx, top_item_cost = top_item_key[1], top_item_key[2]

                objective_with_top_item = self.computeSolutionObjective(curr_solution_items + [top_item_indx])
                top_item_marginal_gain = (objective_with_top_item - self.computeSolutionObjective(curr_solution_items))/top_item_cost

                #Check item now on top - 2nd item on heap
                second_item = self.maxHeap2Guess[0] 
                second_item_heap_gain = second_item[0]*-1

                #If marginal gain of top item is better we add to solution
                if top_item_marginal_gain >= second_item_heap_gain:
                    #Only add if item is within budget
                    if top_item_cost + curr_cost <= self.B:
                        curr_solution_items.append(top_item_indx)
                        curr_objective = objective_with_top_item
                        curr_cost += top_item_cost

                        #Update cost objective map
                        if curr_cost not in cost_objective_map or curr_objective > cost_objective_map[curr_cost][0]:
                            cost_objective_map[curr_cost] = [curr_objective, curr_solution_items.copy()]

                        logging.debug("Adding item {}, curr_objective={:.3f}, curr_cost={}".format(self.items[top_item_indx], curr_objective, curr_cost))
                
                #Otherwise re-insert top item into heap with updated marginal gain
                else:
                    updated_top_item = (top_item_marginal_gain*-1, top_item_indx, top_item_cost)
                    heappush(self.maxHeap2Guess, updated_top_item)

        #Prune cost_objective_map to only keep Pareto optimal solutions
        prunedBudgets, prunedobjectives = [], []
        currentObjective = 0
        for b_prime in sorted(cost_objective_map.keys()):
            if cost_objective_map[b_prime][0] > currentObjective:
                currentObjective = cost_objective_map[b_prime][0]
                prunedBudgets.append(b_prime)
                prunedobjectives.append(currentObjective)
                logging.debug("Approx. Pareto Budget: {}, objective: {}, items: {}".format(b_prime, cost_objective_map[b_prime][0], cost_objective_map[b_prime][1]))

        runTime = time.perf_counter() - startTime
        logging.debug("Prefix Pareto Greedy - 2 Guess Runtime = {:.2f} seconds".format(runTime))

        return prunedBudgets, prunedobjectives, cost_objective_map, runTime
    

    def createmaxHeap1Guess(self, seed_item_index, seed_item_cost):
        '''
        Initialize self.maxHeap1Guess with marginal objective gain for each item that is not the seed
        '''
        #Create max heap to store objectives with respect to new objective function
        self.maxHeap1Guess = []
        heapify(self.maxHeap1Guess)

        #Compute objective with only seed item
        item_objective = self.computeSolutionObjective([seed_item_index])
        
        for i, item_i in enumerate(self.items):
            if i != seed_item_index and (self.costs[i] + seed_item_cost <= self.B): #Only add new items that fit budget

                #Compute marginal objective of new item
                item_marginal_obj = self.computeSolutionObjective([seed_item_index, i]) - item_objective
                item_weight = item_marginal_obj/self.costs[i]

                #push to maxheap - heapItem stored -gain, item index and cost
                heapItem = (item_weight*-1, i, self.costs[i])
                heappush(self.maxHeap1Guess, heapItem)

        return item_objective, seed_item_cost, [seed_item_index]
        

    def oneGuessGreedyPlus(self):
        '''
        1-Guess Greedy+ from Feldman, Nutov, Shoham 2021; Practical Budgeted Submodular Maximization
        '''
        startTime = time.perf_counter()

        #Keep track of all solutions and their costs
        best_sol_items, best_objective, best_cost = [], 0, 0

        #Iterate over all single item seeds
        for i, item_i in enumerate(self.items):
            if self.costs[i] <= self.B:
                #Create priority queue with all other items for this run
                #Initialize variables for this greedy run
                curr_objective, curr_cost, curr_solution_items = self.createmaxHeap1Guess(seed_item_index=i, 
                                                                    seed_item_cost=self.costs[i])
                
                #Assign items greedily using max heap
                #Check if there is an element with cost that fits in budget
                while len(self.maxHeap1Guess) > 0 and (min(key[2] for key in self.maxHeap1Guess) <= (self.B - curr_cost)):
                    
                    #Pop best item from maxHeap and compute marginal gain
                    top_item_key = heappop(self.maxHeap1Guess)
                    top_item_indx, top_item_cost = top_item_key[1], top_item_key[2]

                    objective_with_top_item = self.computeSolutionObjective(curr_solution_items + [top_item_indx])
                    top_item_marginal_gain = (objective_with_top_item - self.computeSolutionObjective(curr_solution_items))/top_item_cost

                    #Check item now on top - 2nd item on heap
                    if len(self.maxHeap1Guess) > 0:
                        second_item = self.maxHeap1Guess[0]
                        second_item_heap_gain = second_item[0]*-1
                    else:
                        second_item_heap_gain = float("-inf")

                    #If marginal gain of top item is better we add to solution
                    if top_item_marginal_gain >= second_item_heap_gain:
                        #Only add if item is within budget
                        if top_item_cost + curr_cost <= self.B:
                            curr_solution_items.append(top_item_indx)
                            curr_objective = objective_with_top_item
                            curr_cost += top_item_cost
                            logging.debug("Adding item {}, curr_objective={:.3f}, curr_cost={}".format(self.items[top_item_indx], curr_objective, curr_cost))
                
                    #Otherwise re-insert top item into heap with updated marginal gain
                    else:
                        updated_top_item = (top_item_marginal_gain*-1, top_item_indx, top_item_cost)
                        heappush(self.maxHeap1Guess, updated_top_item)

                #Store results for run with seed i
                seed_i_objective, seed_i_cost, seed_i_items = curr_objective, curr_cost, curr_solution_items.copy()
                feasible_item_list, feasible_item_cost = [], 0

                #Perform Greedy+ check - Loop over solution in each iteration of plain greedy
                for i, item_i in enumerate(seed_i_items):
                    feasible_item_list.append(item_i)
                    feasible_item_cost += self.costs[item_i]
                    logging.debug("Trying incremental solution:{}, cost:{}".format(feasible_item_list, feasible_item_cost))
                    
                    for j, item_j in enumerate(self.items):
                        #If adding a single item doesn't violate budget
                        if feasible_item_cost + self.costs[j] <= self.B:
                            #Compute objective by adding item to incremental solution
                            added_item_obj = self.computeSolutionObjective(feasible_item_list + [j])
                            
                            #If this solution is better than original solution, store it
                            if added_item_obj > best_objective:
                                seed_i_items = feasible_item_list.copy()
                                seed_i_items.append(j)
                                seed_i_objective = added_item_obj
                                seed_i_cost = feasible_item_cost + self.costs[j]
                                logging.debug("New feasible solution yielded better objective! {}, objective={:.3f}, cost={}".format(seed_i_items,seed_i_objective,seed_i_cost))

                #Keep track of best solution across all seeds
                if seed_i_objective > best_objective:
                    best_sol_items = seed_i_items
                    best_objective = seed_i_objective
                    best_cost = seed_i_cost

        runTime = time.perf_counter() - startTime
        logging.debug("1-Guess Greedy+ Solution:{}, objective:{:.3f}, Cost:{}, Runtime = {:.2f} seconds".format(best_sol_items, best_objective, best_cost, runTime))

        return best_sol_items, best_objective, best_cost, runTime


    def prefixParetoGreedy_1Guess(self):
        '''
        Prefix Pareto Greedy Algorithm - implemented as a variant of 1-Guess Greedy
        '''
        startTime = time.perf_counter()

        #Hashmap to track best objective for each cost
        cost_objective_map = {}

        #Iterate over all single item seeds
        for i, item_i in enumerate(self.items):
            if self.costs[i] <= self.B:
                item_i_obj = self.computeSolutionObjective([i])
                #Update cost objective map
                if self.costs[i] not in cost_objective_map or item_i_obj > cost_objective_map[self.costs[i]][0]:
                    cost_objective_map[self.costs[i]] = [item_i_obj, [i]]

                #Create priority queue with all other items for this run
                #Initialize variables for this greedy run
                curr_objective, curr_cost, curr_solution_items = self.createmaxHeap1Guess(seed_item_index=i, 
                                                                    seed_item_cost=self.costs[i])
                
                #Assign items greedily using max heap
                #Check if there is an element with cost that fits in budget
                while len(self.maxHeap1Guess) > 0 and (min(key[2] for key in self.maxHeap1Guess) <= (self.B - curr_cost)):
                    
                    #Pop best item from maxHeap and compute marginal gain
                    top_item_key = heappop(self.maxHeap1Guess)
                    top_item_indx, top_item_cost = top_item_key[1], top_item_key[2]

                    objective_with_top_item = self.computeSolutionObjective(curr_solution_items + [top_item_indx])
                    top_item_marginal_gain = (objective_with_top_item - self.computeSolutionObjective(curr_solution_items))/top_item_cost

                    #Check item now on top - 2nd item on heap
                    if len(self.maxHeap1Guess) > 0:
                        second_item = self.maxHeap1Guess[0]
                        second_item_heap_gain = second_item[0]*-1
                    else:
                        second_item_heap_gain = float("-inf")

                    #If marginal gain of top item is better we add to solution
                    if top_item_marginal_gain >= second_item_heap_gain:
                        #Only add if item is within budget
                        if top_item_cost + curr_cost <= self.B:
                            curr_solution_items.append(top_item_indx)
                            curr_objective = objective_with_top_item
                            curr_cost += top_item_cost
                        
                            #Update cost objective map
                            if curr_cost not in cost_objective_map or curr_objective > cost_objective_map[curr_cost][0]:
                                cost_objective_map[curr_cost] = [curr_objective, curr_solution_items.copy()]

                            logging.debug("Adding item {}, curr_objective={:.3f}, curr_cost={}".format(self.items[top_item_indx], curr_objective, curr_cost))
                
                    #Otherwise re-insert top item into heap with updated marginal gain
                    else:
                        updated_top_item = (top_item_marginal_gain*-1, top_item_indx, top_item_cost)
                        heappush(self.maxHeap1Guess, updated_top_item)

        #Prune cost_objective_map to only keep Pareto optimal solutions
        prunedBudgets, prunedobjectives = [], []
        currentObjective = 0
        for b_prime in sorted(cost_objective_map.keys()):
            if cost_objective_map[b_prime][0] > currentObjective:
                currentObjective = cost_objective_map[b_prime][0]
                prunedBudgets.append(b_prime)
                prunedobjectives.append(currentObjective)
                logging.debug("Approx. Pareto Budget: {}, objective: {}, items: {}".format(b_prime, cost_objective_map[b_prime][0], cost_objective_map[b_prime][1]))

        runTime = time.perf_counter() - startTime
        logging.debug("Prefix Pareto Greedy - 1 Guess Runtime = {:.2f} seconds".format(runTime))

        return prunedBudgets, prunedobjectives, cost_objective_map, runTime


    def coverage_epsilon_grid(self):
        '''
        Linear objective sweep: for each discrete objective level, find a minimum-cost
        solution using weighted greedy (marginal gain scaled by cost).
        Then prune dominated solutions.
        '''
        startTime = time.perf_counter()

        # Upper bound on objective (using all items)
        max_objective = self.computeSolutionObjective(list(range(self.n)))

        # Minimum objective among single items
        min_objective = min(self.computeSolutionObjective([i]) for i in range(self.n))

        # Discrete objective targets: (1 + epsilon) grid from min to max
        epsilon = 0.1
        target_objectives = []
        curr_obj = min_objective
        while curr_obj < max_objective:
            target_objectives.append(curr_obj)
            curr_obj *= (1 + epsilon)
        target_objectives.append(max_objective)
        logging.info(
            "CoverageLinear epsilon grid: eps=%.3f, min_obj=%.3f, max_obj=%.3f, points=%d",
            epsilon,
            min_objective,
            max_objective,
            len(target_objectives),
        )

        # Track best solution per target objective
        cost_objective_map = {}

        for obj_x in target_objectives:
            # Reset for each target objective
            self.createItemMaxHeap()

            curr_solution_items = []
            curr_objective, curr_cost = 0, 0

            # Weighted greedy until reaching target objective or no feasible item
            while len(self.maxHeap) > 0 and (min(key[2] for key in self.maxHeap) <= (self.B - curr_cost)) and (curr_objective < obj_x):
                top_item_key = heappop(self.maxHeap)
                top_item_indx, top_item_cost = top_item_key[1], top_item_key[2]

                objective_with_top_item = self.computeSolutionObjective(curr_solution_items + [top_item_indx])
                top_item_marginal_gain = (objective_with_top_item - curr_objective) / top_item_cost

                # Compare against next best heap gain
                if len(self.maxHeap) > 0:
                    second_item = self.maxHeap[0]
                    second_item_heap_gain = second_item[0] * -1
                else:
                    second_item_heap_gain = float("-inf")

                if top_item_marginal_gain >= second_item_heap_gain:
                    if top_item_cost + curr_cost <= self.B:
                        curr_solution_items.append(top_item_indx)
                        curr_objective = objective_with_top_item
                        curr_cost += top_item_cost
                        logging.debug("Adding item {}, curr_objective={:.3f}, curr_cost={}".format(self.items[top_item_indx], curr_objective, curr_cost))
                else:
                    updated_top_item = (top_item_marginal_gain * -1, top_item_indx, top_item_cost)
                    heappush(self.maxHeap, updated_top_item)

            # Store if target met within budget
            if curr_objective >= obj_x:
                if obj_x not in cost_objective_map or curr_cost < cost_objective_map[obj_x][0]:
                    cost_objective_map[obj_x] = [curr_cost, curr_solution_items.copy()]

        # Prune dominated solutions: keep strictly increasing objective as cost increases
        prunedBudgets, prunedobjectives = [], []
        pairs = [(data[0], obj) for obj, data in cost_objective_map.items()]
        pairs.sort(key=lambda x: x[0])  # sort by cost
        best_obj = -1.0
        for cost, obj in pairs:
            if obj > best_obj:
                best_obj = obj
                prunedBudgets.append(cost)
                prunedobjectives.append(obj)
                logging.debug("Approx. Pareto Objective: {}, Cost: {}, Items: {}".format(obj, cost, cost_objective_map[obj][1]))

        runTime = time.perf_counter() - startTime
        logging.debug("Coverage Linear Runtime = {:.2f} seconds".format(runTime))

        return prunedBudgets, prunedobjectives, cost_objective_map, runTime


    def cost_epsilon_grid(self):
        '''
        Cost grid sweep: construct a (1 + epsilon) grid from minimum feasible cost
        to maximum budget, run 2-Guess Plain Greedy at each budget, then prune
        dominated solutions by cost.
        '''
        startTime = time.perf_counter()

        # Minimum feasible cost among items
        feasible_costs = [c for c in self.costs if c > 0 and c <= self.B]
        min_cost = min(feasible_costs)

        # Discrete cost targets: (1 + epsilon) grid from min to max budget
        epsilon = 0.1
        target_budgets = []
        curr_cost = min_cost
        while curr_cost < self.B:
            target_budgets.append(curr_cost)
            curr_cost *= (1 + epsilon)
        
        if len(target_budgets) == 0 or target_budgets[-1] < self.B:
            target_budgets.append(self.B)

        logging.info(
            "Cost epsilon grid: eps=%.3f, min_cost=%.3f, max_budget=%.3f, points=%d",
            epsilon,
            min_cost,
            self.B,
            len(target_budgets),
        )

        # Track best objective per realized cost
        cost_objective_map = {}
        total_runtime = 0.0

        original_budget = self.B
        for budgetVal in target_budgets:
            self.B = budgetVal
            sol_items, curr_objective, curr_cost, runTime = self.oneGuessGreedyPlus()
            total_runtime += runTime

            if curr_cost not in cost_objective_map or curr_objective > cost_objective_map[curr_cost][0]:
                cost_objective_map[curr_cost] = [curr_objective, sol_items.copy()]
        
        self.B = original_budget

        # Prune dominated solutions: keep strictly increasing objective as cost increases
        prunedBudgets, prunedobjectives = [], []
        currentObjective = -1.0
        for b_prime in sorted(cost_objective_map.keys()):
            if cost_objective_map[b_prime][0] > currentObjective:
                currentObjective = cost_objective_map[b_prime][0]
                prunedBudgets.append(b_prime)
                prunedobjectives.append(currentObjective)
                logging.debug(
                    "Cost grid Pareto: cost={}, objective={}, items={}".format(
                        b_prime,
                        cost_objective_map[b_prime][0],
                        cost_objective_map[b_prime][1],
                    )
                )

        runTime = time.perf_counter() - startTime
        logging.debug("Cost epsilon grid Runtime = {:.2f} seconds".format(runTime))

        return prunedBudgets, prunedobjectives, cost_objective_map, total_runtime
    

    def FC_Greedy(self):
        '''
        Prune dominated solutions from the union of cost_epsilon_grid and
        coverage_epsilon_grid outputs.
        Args:
            cost_solutions (tuple): (costs, objectives) from cost_epsilon_grid
            coverage_solutions (tuple): (costs, objectives) from coverage_epsilon_grid
        Returns:
            pruned_costs (list): Pareto-optimal costs
            pruned_objectives (list): Pareto-optimal objectives
            total_runtime (float): sum of cost and coverage runtimes
        '''
        cost_costs, cost_objs, cost_objective_map, cost_runtime = self.cost_epsilon_grid()
        cov_costs, cov_objs, cov_objective_map, cov_runtime = self.coverage_epsilon_grid()

        pairs = list(zip(cost_costs, cost_objs)) + list(zip(cov_costs, cov_objs))

        # Sort by cost and keep strictly increasing objective
        pairs.sort(key=lambda x: x[0])
        pruned_costs, pruned_objectives = [], []
        best_obj = float("-inf")
        for cost, obj in pairs:
            if obj > best_obj:
                best_obj = obj
                pruned_costs.append(cost)
                pruned_objectives.append(obj)
                logging.debug("Union Pareto: cost={}, objective={}".format(cost, obj))

        total_runtime = float(cost_runtime) + float(cov_runtime)

        return pruned_costs, pruned_objectives, total_runtime