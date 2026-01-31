import time, json, pickle
from heapq import heappop, heappush, heapify
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

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

    
    def ParetoGreedyDiameter(self):
        """
        Greedy procedure that grows a metric ball around each expert (as center)
        and records the best task coverage achievable at each radius.

        Returns:
            radii                : sorted list of radii considered
            best_coverages       : list of best coverage values
            best_centers         : list of center indices
            best_included_lists  : list of expert index lists
            runtime              : elapsed time in seconds
        """
        startTime = time.perf_counter()

        n = self.n
        task_size = len(self.task)
        task_skills = self.task_skills

        # Map: radius -> (coverage, center, included_list)
        best_at_radius = {}

        # Pre-sort neighbors for each center
        neighbor_order = np.argsort(self.pairwise_costs, axis=1)
        neighbor_dists = np.take_along_axis(self.pairwise_costs, neighbor_order, axis=1)

        for center in range(n):
            covered_skills = set()
            included = []

            for idx in range(n):
                u = neighbor_order[center, idx]
                r = neighbor_dists[center, idx]

                # Add new expert
                covered_skills |= set(self.experts[u])
                included.append(u)

                # Compute coverage
                cov = len(covered_skills & task_skills) / task_size

                # Update best solution for this radius
                if r not in best_at_radius or cov > best_at_radius[r][0]:
                    best_at_radius[r] = (cov, center, included.copy())

                # Optional early stop if full coverage achieved
                if cov >= 1.0:
                    break

        # Pareto pruning (increasing radius)
        radii = sorted(best_at_radius.keys())
        best_radii = []
        best_coverages = []
        best_centers = []
        best_included_lists = []

        best_so_far = -1
        for r in radii:
            cov, center, included = best_at_radius[r]
            if cov > best_so_far:
                best_so_far = cov
                best_radii.append(r)
                best_coverages.append(cov)
                best_centers.append(center)
                best_included_lists.append(included)

        runTime = time.perf_counter() - startTime
        logging.info("GreedyThresholdDiameter finished: max_coverage={:.3f}, runtime={:.3f}s"
            .format(max(best_coverages) if best_coverages else 0.0, runTime))

        return best_radii, best_coverages, best_centers, best_included_lists, runTime
    

    def graphPruning(self):
        '''
        Baseline algorithm starting with the entire graph, prune nodes in greedy heuristic manner
        Keep track of the coverage objective and diameter cost at each step
        '''
        startTime = time.perf_counter()

        n = self.n
        task_skills = self.task_skills
        task_size = len(self.task)

        # Precompute expert skill sets
        expert_skill_sets = [set(e) for e in self.experts]

        # Initialize included experts and skill counts
        included = list(range(n))
        skill_counts = {}
        for idx in included:
            for s in expert_skill_sets[idx]:
                if s in task_skills:
                    skill_counts[s] = skill_counts.get(s, 0) + 1

        def compute_coverage(counts):
            if task_size == 0:
                return 0.0
            covered = sum(1 for s, c in counts.items() if c > 0)
            return covered / task_size

        def compute_diameter(indices):
            if len(indices) <= 1:
                return 0.0
            sub = self.pairwise_costs[np.ix_(indices, indices)]
            return float(np.max(sub))

        # Track sequence of (diameter, coverage, included_list)
        seq_diams = []
        seq_covs = []
        seq_included = []

        # Greedy pruning loop
        while len(included) > 0:
            curr_cov = compute_coverage(skill_counts)
            curr_diam = compute_diameter(included)
            seq_diams.append(curr_diam)
            seq_covs.append(curr_cov)
            seq_included.append(included.copy())

            if len(included) == 1:
                break

            # Compute unique contribution for each expert
            unique_contrib = {}
            for idx in included:
                count = 0
                for s in expert_skill_sets[idx]:
                    if s in task_skills and skill_counts.get(s, 0) == 1:
                        count += 1
                unique_contrib[idx] = count

            # Break ties by removing node with largest max distance to others
            best_remove = None
            best_key = None
            for idx in included:
                max_dist = float(np.max(self.pairwise_costs[idx, included]))
                key = (unique_contrib[idx], -max_dist)
                if best_key is None or key < best_key:
                    best_key = key
                    best_remove = idx

            # Remove selected expert and update skill counts
            if best_remove is None:
                break
            included.remove(best_remove)
            for s in expert_skill_sets[best_remove]:
                if s in task_skills:
                    skill_counts[s] = max(0, skill_counts.get(s, 0) - 1)

        # Map: diameter -> (coverage, center, included_list)
        best_at_diameter = {}
        for diam, cov, incl in zip(seq_diams, seq_covs, seq_included):
            if diam not in best_at_diameter or cov > best_at_diameter[diam][0]:
                best_at_diameter[diam] = (cov, -1, incl)

        # Pareto pruning (increasing diameter)
        radii = sorted(best_at_diameter.keys())
        best_coverages = []
        best_centers = []
        best_included_lists = []

        best_so_far = -1
        for r in radii:
            cov, center, incl = best_at_diameter[r]
            if cov > best_so_far:
                best_so_far = cov
                best_coverages.append(cov)
                best_centers.append(center)
                best_included_lists.append(incl)

        runTime = time.perf_counter() - startTime
        logging.info(
            "GraphPruning finished: max_coverage={:.3f}, runtime={:.3f}s".format(
                max(best_coverages) if best_coverages else 0.0, runTime
            )
        )

        return radii, best_coverages, best_centers, best_included_lists, runTime


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


    # def greedyThresholdDiameter(self):
    #     '''
    #     Greedy procedure that grows a ball (by radius) around every expert (as center)
    #     and tracks the best task coverage achievable at each radius.

    #     Returns:
    #         radii                : sorted 1D numpy array of radii considered (includes 0)
    #         best_coverages       : list of best coverage values (same length as radii)
    #         best_centers         : list of center expert indices that achieved the best coverage
    #         best_included_lists  : list of lists of expert indices included for the best center at each radius
    #         runtime              : elapsed time in seconds
    #     '''
    #     startTime = time.perf_counter()

    #     #get unique pairwise costs
    #     radii = np.insert(np.unique(self.pairwise_costs), 0, 0.0)  
    #     best_coverages, best_centers, best_included_lists = [], [], []

    #     #Maintain sets of experts centered at each expert
    #     expertBalls = {i: set(self.experts[i]) for i in range(self.n)}
    #     expertBallsIndices = {i: [i] for i in range(self.n)}

    #     # Precompute per-center neighbor order and corresponding distances (increasing)
    #     neighbor_order = np.argsort(self.pairwise_costs, axis=1, kind='mergesort')   # shape (n,n)
    #     neighbor_dists = np.take_along_axis(self.pairwise_costs, neighbor_order, axis=1)  

    #     r_prev = 0

    #     # For each radius, evaluate ball around every expert and keep the best
    #     for r in radii:
    #         best_cov, best_center, best_included = -1, None, []

    #         for center in range(self.n):
    #             # find indices in the sorted neighbor_dists[center] that lie in (r_prev, r]
    #             # prev_idx = count of distances <= r_prev; end_idx = count of distances <= r
    #             prev_idx = np.searchsorted(neighbor_dists[center], r_prev, side='right')
    #             end_idx = np.searchsorted(neighbor_dists[center], r, side='right')

    #             # neighbors added at this step: those between prev_idx (exclusive) and end_idx (inclusive)
    #             if end_idx > prev_idx:
    #                 added_indices = neighbor_order[center, prev_idx:end_idx].tolist()
    #             else:
    #                 added_indices = []

    #             # update ball sets and index lists incrementally
    #             if added_indices:
    #                 # union all skills of included experts
    #                 expertBalls[center] = expertBalls[center].union(*(set(self.experts[j]) for j in added_indices))
    #                 expertBallsIndices[center].extend(added_indices)

    #             # compute coverage 
    #             cov = (len(expertBalls[center] & self.task_skills) / len(self.task))

    #             if cov > best_cov:
    #                 best_cov = cov
    #                 best_center = center
    #                 best_included = expertBallsIndices[center]

    #             # early stop if full coverage reached
    #             if best_cov >= 1.0:
    #                 break

    #         best_coverages.append(best_cov)
    #         best_centers.append(best_center)
    #         best_included_lists.append(best_included)
    #         r_prev = r #update previous radius
            
    #         # early stop if full coverage reached
    #         if best_cov >= 1.0:
    #             radii = radii[:len(best_coverages)]  # trim radii to current length
    #             break

    #     runTime = time.perf_counter() - startTime
    #     logging.info("GreedyThresholdDiameter finished: max_coverage={:.3f}, runtime={:.3f}s".format(max(best_coverages) if best_coverages else 0.0, runTime))

    #     return radii, best_coverages, best_centers, best_included_lists, runTime