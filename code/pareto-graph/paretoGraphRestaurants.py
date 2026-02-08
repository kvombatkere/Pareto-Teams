import time
import pickle
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import logging

logging.basicConfig(format='%(asctime)s |%(levelname)s: %(message)s', level=logging.INFO)


class paretoGraphRestaurants:
    '''
    Define a class for restaurant recommendations with graph (diameter) cost.
    Objective uses facility-location style coverage based on similarity matrix.
    '''

    def __init__(self, n_items, simMatrix, pairwise_costs=None):
        '''
        Initialize instance with items and similarity matrix.
        ARGS:
            n_items         : list of n items; each item is a restaurant
            simMatrix       : similarity matrix between items (shape n x n)
            pairwise_costs  : optional distance matrix (shape n x n). If None, uses 1 - simMatrix.
        '''
        self.items = n_items
        self.simMatrix = np.asarray(simMatrix, dtype=float)
        self.n = len(self.items)
        if pairwise_costs is None:
            self.pairwise_costs = self.sim_to_distance(self.simMatrix)
        else:
            self.pairwise_costs = np.asarray(pairwise_costs, dtype=float)
        logging.info("Initialized Pareto Restaurants - Graph Cost Instance, Num Items:{}".format(self.n))

    @staticmethod
    def sim_to_distance(sim_matrix):
        dist = 1.0 - np.asarray(sim_matrix, dtype=float)
        np.fill_diagonal(dist, 0.0)
        return np.clip(dist, 0.0, None)

    @staticmethod
    def _compute_diameter_from_indices(pairwise_costs, indices):
        if len(indices) <= 1:
            return 0.0
        sub = pairwise_costs[np.ix_(indices, indices)]
        off_diag = sub[~np.eye(len(indices), dtype=bool)]
        if off_diag.size == 0:
            return 0.0
        return float(np.max(off_diag))

    def _objective_from_max_sims(self, max_sims):
        if self.n == 0:
            return 0.0
        return float(np.sum(max_sims))

    def _compute_max_sims_for_indices(self, indices):
        if len(indices) == 0 or self.n == 0:
            return np.zeros(self.n, dtype=float), 0.0
        sims_sub = self.simMatrix[:, indices]
        max_sims = np.max(sims_sub, axis=1)
        return max_sims, self._objective_from_max_sims(max_sims)

    def ParetoGreedyDiameter(self):
        """
        Greedy procedure that grows a metric ball around each item (as center)
        and records the best objective achievable at each radius.

        Returns:
            diameters            : list of diameters considered
            best_objectives      : list of best objective values
            best_centers         : list of center indices
            best_included_lists  : list of item index lists
            runtime              : elapsed time in seconds
        """
        startTime = time.perf_counter()
        n = self.n

        best_at_radius = {}

        neighbor_order = np.argsort(self.pairwise_costs, axis=1)
        neighbor_dists = np.take_along_axis(self.pairwise_costs, neighbor_order, axis=1)

        for center in range(n):
            included = []
            max_sims = np.zeros(n, dtype=float)

            for idx in range(n):
                u = neighbor_order[center, idx]
                r = neighbor_dists[center, idx]

                included.append(u)
                max_sims = np.maximum(max_sims, self.simMatrix[:, u])
                obj = self._objective_from_max_sims(max_sims)

                if r not in best_at_radius or obj > best_at_radius[r][0]:
                    best_at_radius[r] = (obj, center, included.copy())

        radii = sorted(best_at_radius.keys())
        best_diams, best_objectives, best_centers, best_included_lists = [], [], [], []

        best_so_far = -1
        for r in radii:
            obj, center, included = best_at_radius[r]
            if obj > best_so_far:
                best_so_far = obj
                best_diams.append(r)
                best_objectives.append(obj)
                best_centers.append(center)
                best_included_lists.append(included)

        runTime = time.perf_counter() - startTime
        num_items_chosen = len(best_included_lists[-1]) if best_included_lists else 0
        logging.info("ParetoGreedyDiameter finished: max_objective={:.3f}, runtime={:.3f}s, items={}".format(
            max(best_objectives) if best_objectives else 0.0, runTime, num_items_chosen
        ))

        best_diameters = [self._compute_diameter_from_indices(self.pairwise_costs, incl)
                          for incl in best_included_lists]

        return best_diameters, best_objectives, best_centers, best_included_lists, runTime

    def graphPruning(self):
        '''
        Baseline algorithm starting with the entire graph, prune items in greedy heuristic manner
        and track objective and diameter at each step.
        '''
        startTime = time.perf_counter()

        n = self.n
        included = list(range(n))

        seq_diams, seq_objs, seq_included = [], [], []

        while len(included) > 0:
            _, curr_obj = self._compute_max_sims_for_indices(included)
            curr_diam = self._compute_diameter_from_indices(self.pairwise_costs, included)
            seq_diams.append(curr_diam)
            seq_objs.append(curr_obj)
            seq_included.append(included.copy())

            if len(included) == 1:
                break

            sims_sub = self.simMatrix[:, included]
            top1_idx = np.argmax(sims_sub, axis=1)
            top1_vals = sims_sub[np.arange(n), top1_idx]
            if len(included) > 1:
                sims_sub_copy = sims_sub.copy()
                sims_sub_copy[np.arange(n), top1_idx] = -np.inf
                top2_vals = np.max(sims_sub_copy, axis=1)
                top2_vals = np.where(np.isfinite(top2_vals), top2_vals, 0.0)
            else:
                top2_vals = np.zeros(n, dtype=float)

            losses = np.zeros(len(included), dtype=float)
            delta = top1_vals - top2_vals
            losses += np.bincount(top1_idx, weights=delta, minlength=len(included))

            best_remove = None
            best_key = None
            for pos, idx in enumerate(included):
                max_dist = float(np.max(self.pairwise_costs[idx, included]))
                key = (losses[pos], -max_dist)
                if best_key is None or key < best_key:
                    best_key = key
                    best_remove = idx

            if best_remove is None:
                break
            included.remove(best_remove)

        best_at_diameter = {}
        for diam, obj, incl in zip(seq_diams, seq_objs, seq_included):
            if diam not in best_at_diameter or obj > best_at_diameter[diam][0]:
                best_at_diameter[diam] = (obj, -1, incl)

        radii = sorted(best_at_diameter.keys())
        best_objectives, best_centers, best_included_lists = [], [], []
        best_so_far = -1
        for r in radii:
            obj, center, incl = best_at_diameter[r]
            if obj > best_so_far:
                best_so_far = obj
                best_objectives.append(obj)
                best_centers.append(center)
                best_included_lists.append(incl)

        runTime = time.perf_counter() - startTime
        logging.info("GraphPruning finished: max_objective={:.3f}, runtime={:.3f}s".format(
            max(best_objectives) if best_objectives else 0.0, runTime))

        return radii, best_objectives, best_centers, best_included_lists, runTime


    def plainGreedyDistanceScaled(self):
        '''
        Plain Greedy baseline that scales marginal objective gain by the
        average distance to nodes in the current solution.
        '''
        startTime = time.perf_counter()

        n = self.n
        max_obj = float(n)
        included = []
        max_sims = np.zeros(n, dtype=float)

        seq_diams, seq_objs, seq_included = [], [], []

        remaining = set(range(n))
        while remaining:
            best_idx = None
            best_score = -1
            best_obj = None
            curr_obj = self._objective_from_max_sims(max_sims)

            for idx in remaining:
                new_max = np.maximum(max_sims, self.simMatrix[:, idx])
                obj = self._objective_from_max_sims(new_max)
                marginal_gain = obj - curr_obj

                if included:
                    avg_dist = float(np.mean(self.pairwise_costs[idx, included]))
                else:
                    avg_dist = 1.0

                score = marginal_gain if avg_dist <= 0 else marginal_gain / avg_dist

                if score > best_score:
                    best_score = score
                    best_idx = idx
                    best_obj = obj

            if best_idx is None:
                break

            included.append(best_idx)
            max_sims = np.maximum(max_sims, self.simMatrix[:, best_idx])
            remaining.remove(best_idx)

            curr_diam = self._compute_diameter_from_indices(self.pairwise_costs, included)
            seq_diams.append(curr_diam)
            seq_objs.append(best_obj if best_obj is not None else 0.0)
            seq_included.append(included.copy())

            if max_obj > 0 and best_obj is not None and best_obj >= 0.999 * max_obj:
                break

        runTime = time.perf_counter() - startTime
        logging.info(
            "PlainGreedyDistanceScaled finished: max_objective={:.3f}, runtime={:.3f}s".format(
                max(seq_objs) if seq_objs else 0.0, runTime
            )
        )

        return seq_diams, seq_objs, [-1 for _ in seq_diams], seq_included, runTime

    def topKDegree(self):
        '''
        Top-K baseline: compute a single ordering by highest degree
        (sum of edge weights), then add nodes in that order and track the
        resulting diameter after each addition.
        '''
        startTime = time.perf_counter()

        n = self.n
        degrees = np.sum(self.pairwise_costs, axis=1)
        if n > 0:
            degrees = degrees - np.diag(self.pairwise_costs)
        degrees = np.asarray(degrees, dtype=float)
        ordered_indices = list(np.argsort(-degrees, kind='mergesort'))

        seq_diams, seq_objs, seq_included = [], [], []
        included = []
        max_sims = np.zeros(n, dtype=float)

        last_diam = None
        for idx in ordered_indices:
            included.append(idx)
            max_sims = np.maximum(max_sims, self.simMatrix[:, idx])
            curr_diam = self._compute_diameter_from_indices(self.pairwise_costs, included)
            curr_obj = self._objective_from_max_sims(max_sims)
            if last_diam is None or curr_diam != last_diam:
                seq_diams.append(curr_diam)
                seq_objs.append(curr_obj)
                seq_included.append(included.copy())
                last_diam = curr_diam
            else:
                # Same diameter: update objective for this diameter step
                seq_objs[-1] = curr_obj
                seq_included[-1] = included.copy()

        runTime = time.perf_counter() - startTime
        logging.info(
            "TopKDistanceScaled finished: max_objective={:.3f}, runtime={:.3f}s".format(
                max(seq_objs) if seq_objs else 0.0, runTime
            )
        )

        return seq_diams, seq_objs, [-1 for _ in seq_diams], seq_included, runTime


def import_yelp_data(data_path_prefix):
    '''
    Import Yelp dataset from pickled data files.
    Expects files: {prefix}ids.pkl and {prefix}sim.pkl
    '''
    with open(data_path_prefix + 'ids.pkl', "rb") as fp:
        item_ids = pickle.load(fp)
        logging.info("Imported Yelp ids, Num Items: {}".format(len(item_ids)))

    with open(data_path_prefix + 'sim.pkl', "rb") as fp:
        simMatrix = pickle.load(fp)
        logging.info("Imported Yelp similarity matrix, Shape: {}".format(np.asarray(simMatrix).shape))

    return item_ids, simMatrix
