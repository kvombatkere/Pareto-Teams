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
            task                : task to be accomplished;
            n_experts           : list of n experts; each expert is a list of skills
            pairwise_costs      : pairwise cost matrix between experts
            size_univ           : number of distinct skills in the universe
            budget              : knapsack budget
        '''
        self.task = task
        self.task_skills = set(task)

        self.experts = n_experts
        self.m, self.n = size_univ, len(self.experts)
        self.pairwise_costs = np.asarray(pairwise_costs, dtype=float)
        self.B = budget
        logging.info("Initialized Pareto Coverage - Graph Cost Instance, Task:{}, Num Experts:{}, Budget={}".format(self.task, self.n, self.B))


    def greedyThresholdDiameter(self):
        '''
        Greedy procedure that grows a ball (by radius) around every expert (as center)
        and tracks the best task coverage achievable at each radius.

        Returns:
            radii                : sorted 1D numpy array of radii considered (includes 0)
            best_coverages       : list of best coverage values (same length as radii)
            best_centers         : list of center expert indices that achieved the best coverage
            best_included_lists  : list of lists of expert indices included for the best center at each radius
            runtime              : elapsed time in seconds
        '''
        startTime = time.perf_counter()

        #get unique pairwise costs
        radii = np.insert(np.unique(self.pairwise_costs), 0, 0.0)  
        best_coverages, best_centers, best_included_lists = [], [], []

        #Maintain sets of experts centered at each expert
        expertBalls = {i: set(self.experts[i]) for i in range(self.n)}
        expertBallsIndices = {i: [i] for i in range(self.n)}

        # Precompute per-center neighbor order and corresponding distances (increasing)
        neighbor_order = np.argsort(self.pairwise_costs, axis=1, kind='mergesort')   # shape (n,n)
        neighbor_dists = np.take_along_axis(self.pairwise_costs, neighbor_order, axis=1)  

        r_prev = 0

        # For each radius, evaluate ball around every expert and keep the best
        for r in radii:
            best_cov, best_center, best_included = -1, None, []

            for center in range(self.n):
                # find indices in the sorted neighbor_dists[center] that lie in (r_prev, r]
                # prev_idx = count of distances <= r_prev; end_idx = count of distances <= r
                prev_idx = np.searchsorted(neighbor_dists[center], r_prev, side='right')
                end_idx = np.searchsorted(neighbor_dists[center], r, side='right')

                # neighbors added at this step: those between prev_idx (exclusive) and end_idx (inclusive)
                if end_idx > prev_idx:
                    added_indices = neighbor_order[center, prev_idx:end_idx].tolist()
                else:
                    added_indices = []

                # update ball sets and index lists incrementally
                if added_indices:
                    # union all skills of included experts
                    expertBalls[center] = expertBalls[center].union(*(set(self.experts[j]) for j in added_indices))
                    expertBallsIndices[center].extend(added_indices)

                # compute coverage 
                cov = (len(expertBalls[center] & self.task_skills) / len(self.task))

                if cov > best_cov:
                    best_cov = cov
                    best_center = center
                    best_included = expertBallsIndices[center]

                # early stop if full coverage reached
                if best_cov >= 1.0:
                    break

            best_coverages.append(best_cov)
            best_centers.append(best_center)
            best_included_lists.append(best_included)
            r_prev = r #update previous radius
            
            # early stop if full coverage reached
            if best_cov >= 1.0:
                radii = radii[:len(best_coverages)]  # trim radii to current length
                break

        runTime = time.perf_counter() - startTime
        logging.info("GreedyThresholdDiameter finished: max_coverage={:.3f}, runtime={:.3f}s".format(max(best_coverages) if best_coverages else 0.0, runTime))

        return radii, best_coverages, best_centers, best_included_lists, runTime