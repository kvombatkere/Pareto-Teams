import time, json, pickle
from heapq import heappop, heappush, heapify
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import logging
logging.basicConfig(format='%(asctime)s |%(levelname)s: %(message)s', level=logging.INFO)

class paretoCardinalityRestaurants():
    '''
    Define a class for restaurant recommendations with cardinality constraint
    '''

    def __init__(self, n_items, simMatrix, k_max):
        '''
        Initialize instance with n items, similarity matrix and cardinality constraint
        ARGS:
            n_items     : list of n items; each item is a restaurant
            simMatrix   : similarity matrix between items
            k_max       : cardinality constraint
        '''
        self.items = n_items
        self.simMatrix = simMatrix
        self.n = len(self.items)
        self.k_max = k_max
        self.kSolDict = {}
        logging.info("Initialized Pareto Restaurant - Cardinality Constraint Instance, Num Items:{}, k_max={}".format(self.n, k_max))


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

            #push to maxheap - heapItem stored -gain, item index
            heapItem = (marginal_gain*-1, e)
            heappush(self.maxHeap, heapItem)

        return 


    def greedyCardinality(self):
        '''
        Greedy Algorithm for Submodular Maximization with cardinality constraint
        '''
        startTime = time.perf_counter()

        #Solution items and current objective
        curr_solution_items = []
        curr_objective = 0
        k_val = 0

        #Create maxheap with objectives
        self.createItemMaxHeap()

        #Assign items greedily using max heap
        while len(self.maxHeap) > 1 and (len(curr_solution_items) < self.k_max):
            
            #Pop best item from maxHeap and compute marginal gain
            top_item_key = heappop(self.maxHeap)
            top_item_indx = top_item_key[1]

            objective_with_top_item = self.computeSolutionObjective(curr_solution_items + [top_item_indx])
            top_item_marginal_gain = objective_with_top_item - curr_objective

            #Check item now on top - 2nd item on heap
            second_item = self.maxHeap[0] 
            second_item_heap_gain = second_item[0]*-1

            #If marginal gain of top item is better we add to solution
            if top_item_marginal_gain >= second_item_heap_gain:
                #Only add if within cardinality
                if len(curr_solution_items) < self.k_max:
                    k_val += 1
                    curr_solution_items.append(top_item_indx)
                    curr_objective = objective_with_top_item
                    self.kSolDict[k_val] = {"Items": [self.items[idx] for idx in curr_solution_items], "Skills": curr_solution_items, "Coverage": curr_objective}
                    logging.debug("k = {}, Adding item {}, curr_objective={:.3f}".format(k_val, self.items[top_item_indx], curr_objective))
            
            #Otherwise re-insert top item into heap with updated marginal gain
            else:
                updated_top_item = (top_item_marginal_gain*-1, top_item_indx)
                heappush(self.maxHeap, updated_top_item)

        runTime = time.perf_counter() - startTime
        logging.info("Cardinality Greedy Solution:{}, Objective:{:.3f}, Runtime = {:.2f} seconds".format(curr_solution_items, curr_objective, runTime))

        return [self.items[idx] for idx in curr_solution_items], curr_solution_items, curr_objective, runTime


    def top_k(self):
        '''
        Top-k Algorithm: Select the top k items from the heap without updates.
        Marginal gains are computed w.r.t. the empty set (i.e., individual gains).
        '''
        startTime = time.perf_counter()

        #Solution items and current objective
        curr_solution_items = []
        curr_objective = 0

        #Create maxheap with gains
        self.createItemMaxHeap()

        #Select top k items
        for k_val in range(1, self.k_max + 1):
            if self.maxHeap:
                top_item_key = heappop(self.maxHeap)
                top_item_indx = top_item_key[1]

                curr_solution_items.append(top_item_indx)
                curr_objective = self.computeSolutionObjective(curr_solution_items)
                self.kSolDict[k_val] = {"Items": [self.items[idx] for idx in curr_solution_items], "Skills": curr_solution_items, "Coverage": curr_objective}
                logging.debug("k = {}, Adding top item {}, curr_objective={:.3f}".format(k_val, self.items[top_item_indx], curr_objective))

        runTime = time.perf_counter() - startTime
        logging.info("Top-k Solution:{}, Objective:{:.3f}, Runtime = {:.2f} seconds".format(curr_solution_items, curr_objective, runTime))

        return [self.items[idx] for idx in curr_solution_items], curr_solution_items, curr_objective, runTime


    def random_selection(self):
        '''
        Random Algorithm: Randomly select k distinct items.
        '''
        startTime = time.perf_counter()

        #Randomly select k_max distinct item indices
        selected_indices = np.random.choice(self.n, size=self.k_max, replace=False)

        #Solution items and current objective
        curr_solution_items = []

        for k_val in range(1, self.k_max + 1):
            curr_solution_items.append(selected_indices[k_val-1])
            curr_objective = self.computeSolutionObjective(curr_solution_items)
            self.kSolDict[k_val] = {"Items": [self.items[idx] for idx in curr_solution_items], "Skills": curr_solution_items, "Coverage": curr_objective}

        runTime = time.perf_counter() - startTime
        logging.info("Random Selection Solution:{}, Objective:{:.3f}, Runtime = {:.2f} seconds".format(curr_solution_items, curr_objective, runTime))

        return [self.items[idx] for idx in curr_solution_items], curr_solution_items, curr_objective, runTime