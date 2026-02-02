import time
import pickle
import numpy as np
import networkx as nx
from collections import defaultdict
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import logging

logging.basicConfig(format='%(asctime)s |%(levelname)s: %(message)s', level=logging.INFO)


class paretoGraphInfluence:
    '''
    Define a class for influence maximization with graph (diameter) cost.
    Objective is expected influence spread (normalized by number of nodes).
    '''

    def __init__(self, G, pairwise_costs=None, nodes=None, num_samples=35, graph_samples=None):
        '''
        Initialize instance with graph and optional pairwise costs.
        ARGS:
            G               : networkx graph
            pairwise_costs  : optional distance matrix (shape n x n). If None, computed via shortest paths.
            nodes           : optional list of nodes (ordering to match pairwise_costs)
            num_samples     : number of graph samples for Monte Carlo
            graph_samples   : pre-computed graph samples (optional)
        '''
        self.G = G
        self.nodes = list(G.nodes()) if nodes is None else list(nodes)
        self.n = len(self.nodes)
        self.num_samples = num_samples
        if pairwise_costs is None:
            self.pairwise_costs, self.nodes = compute_pairwise_costs_from_graph(G, self.nodes)
        else:
            self.pairwise_costs = np.asarray(pairwise_costs, dtype=float)
        if graph_samples is not None:
            self.graph_samples = graph_samples
        else:
            self.initialize_graph_samples()
        logging.info("Initialized Pareto Influence - Graph Cost Instance, Num Nodes:{}".format(self.n))

    @staticmethod
    def _compute_diameter_from_indices(pairwise_costs, indices):
        if len(indices) <= 1:
            return 0.0
        sub = pairwise_costs[np.ix_(indices, indices)]
        off_diag = sub[~np.eye(len(indices), dtype=bool)]
        if off_diag.size == 0:
            return 0.0
        return float(np.max(off_diag))

    def initialize_graph_samples(self):
        '''
        Initialize Monte Carlo samples of the graph under independent cascade model
        '''
        self.graph_samples = []
        for _ in range(self.num_samples):
            G_sample = nx.Graph()
            connected_components = defaultdict()
            for u, v, data in self.G.edges(data=True):
                success = np.random.uniform(0, 1)
                if success < data.get('weight', 0.1):
                    G_sample.add_edge(u, v)
            for c in nx.connected_components(G_sample):
                for node in c:
                    connected_components[node] = c
            self.graph_samples.append(connected_components)

    def compute_influence(self, node_list):
        '''
        Compute expected influence spread of a set of nodes (unnormalized).
        '''
        if self.n == 0:
            return 0.0
        if not node_list:
            return 0.0
        spread = []
        for sample in self.graph_samples:
            connected_components = sample
            active_nodes = set()
            for node in node_list:
                component = connected_components.get(node)
                if component is not None:
                    active_nodes.update(component)
            spread.append(len(active_nodes))
        return float(np.mean(spread))

    def compute_marginal_gain(self, current_nodes, new_node):
        '''
        Compute marginal gain of adding new_node to current_nodes.
        '''
        if self.n == 0:
            return 0.0
        if not self.graph_samples:
            return 0.0
        current_nodes = list(current_nodes)
        marginal_gain = 0.0
        for sample in self.graph_samples:
            connected_components = sample
            active_nodes = set()
            for node in current_nodes:
                component = connected_components.get(node)
                if component is not None:
                    active_nodes.update(component)
            component_new = connected_components.get(new_node)
            if component_new is not None:
                marginal_gain += len(component_new - active_nodes)
        return float(marginal_gain / len(self.graph_samples))

    def ParetoGreedyDiameter(self):
        """
        Greedy procedure that grows a metric ball around each node (as center)
        and records the best influence achievable at each radius.
        """
        startTime = time.perf_counter()
        n = self.n

        best_at_radius = {}

        neighbor_order = np.argsort(self.pairwise_costs, axis=1)
        neighbor_dists = np.take_along_axis(self.pairwise_costs, neighbor_order, axis=1)

        for center in range(n):
            included = []
            active_nodes = [set() for _ in range(len(self.graph_samples))]

            for idx in range(n):
                u = neighbor_order[center, idx]
                r = neighbor_dists[center, idx]
                node_u = self.nodes[u]

                included.append(u)

                total_active = 0
                for s_idx, sample in enumerate(self.graph_samples):
                    component = sample.get(node_u)
                    if component is not None:
                        active_nodes[s_idx].update(component)
                    total_active += len(active_nodes[s_idx])

                obj = float(total_active / max(1, len(self.graph_samples)))

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
        best_diameters = [self._compute_diameter_from_indices(self.pairwise_costs, incl)
                          for incl in best_included_lists]

        logging.info("GreedyThresholdDiameter finished: max_influence={:.3f}, runtime={:.3f}s".format(
            max(best_objectives) if best_objectives else 0.0, runTime
        ))
        logging.info(
            "GreedyThresholdDiameter solutions: {}".format(
                "; ".join(
                    f"(d={d:.3f}, infl={v:.3f}, |S|={len(s)})"
                    for d, v, s in zip(best_diameters, best_objectives, best_included_lists)
                )
            )
        )

        return best_diameters, best_objectives, best_centers, best_included_lists, runTime

    def graphPruning(self):
        '''
        Baseline algorithm starting with the entire graph, prune nodes in greedy heuristic manner
        and track influence and diameter at each step.
        '''
        startTime = time.perf_counter()

        n = self.n
        included = list(range(n))

        seq_diams, seq_objs, seq_included = [], [], []

        while len(included) > 0:
            curr_nodes = [self.nodes[i] for i in included]
            curr_obj = self.compute_influence(curr_nodes)
            curr_diam = self._compute_diameter_from_indices(self.pairwise_costs, included)
            seq_diams.append(curr_diam)
            seq_objs.append(curr_obj)
            seq_included.append(included.copy())

            if len(included) == 1:
                break

            best_remove = None
            best_key = None
            for idx in included:
                remaining_nodes = [self.nodes[i] for i in included if i != idx]
                new_obj = self.compute_influence(remaining_nodes)
                loss = curr_obj - new_obj
                max_dist = float(np.max(self.pairwise_costs[idx, included]))
                key = (loss, -max_dist)
                if best_key is None or key < best_key:
                    best_key = key
                    best_remove = idx

            if best_remove is None:
                break
            included.remove(best_remove)

        best_at_diameter = {}
        for diam, obj, incl in zip(seq_diams, seq_objs, seq_included):
            if diam not in best_at_diameter or obj > best_at_diameter[diam][0]:
                best_at_diameter[diam] = (obj, incl)

        diameters = []
        best_objectives = []
        best_included_lists = []
        for diam in sorted(best_at_diameter.keys()):
            obj, incl = best_at_diameter[diam]
            diameters.append(diam)
            best_objectives.append(obj)
            best_included_lists.append(incl)

        best_centers = [-1 for _ in diameters]

        runTime = time.perf_counter() - startTime
        logging.info(
            "GraphPruning finished: max_influence={:.3f}, runtime={:.3f}s".format(
                max(best_objectives) if best_objectives else 0.0, runTime
            )
        )
        logging.info(
            "GraphPruning solutions: {}".format(
                "; ".join(
                    f"(d={d:.3f}, infl={v:.3f}, |S|={len(s)})"
                    for d, v, s in zip(diameters, best_objectives, best_included_lists)
                )
            )
        )

        return diameters, best_objectives, best_centers, best_included_lists, runTime

    def plainGreedyDistanceScaled(self, diameters=None, num_steps=8):
        '''
        Plain Greedy baseline that scales marginal influence gain by the
        average distance to nodes in the current solution.

        If diameters is provided, greedily build a solution that satisfies
        each diameter constraint and return one solution per diameter.
        '''
        startTime = time.perf_counter()

        n = self.n
        max_obj = float(n)

        if diameters is None:
            max_diameter = float(np.max(self.pairwise_costs)) if self.pairwise_costs.size > 0 else 1.0
            min_diameter = 0.0
            diameters = list(np.linspace(min_diameter, max_diameter, num_steps))
        else:
            diameters = list(diameters)

        if len(diameters) == 0:
            included = []
            seq_diams, seq_objs, seq_included = [], [], []

            remaining = set(range(n))
            while remaining:
                best_idx = None
                best_score = -1
                best_obj = None
                curr_nodes = [self.nodes[i] for i in included]
                curr_obj = self.compute_influence(curr_nodes)

                for idx in remaining:
                    node_idx = self.nodes[idx]
                    obj = self.compute_influence(curr_nodes + [node_idx])
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
                remaining.remove(best_idx)

                curr_diam = self._compute_diameter_from_indices(self.pairwise_costs, included)
                seq_diams.append(curr_diam)
                seq_objs.append(best_obj if best_obj is not None else 0.0)
                seq_included.append(included.copy())

                if max_obj > 0 and best_obj is not None and best_obj >= 0.999 * max_obj:
                    break

            runTime = time.perf_counter() - startTime
            logging.info(
                "PlainGreedyDistanceScaled finished: max_influence={:.3f}, runtime={:.3f}s".format(
                    max(seq_objs) if seq_objs else 0.0, runTime
                )
            )
            logging.info(
                "PlainGreedyDistanceScaled solutions: {}".format(
                    "; ".join(
                        f"(d={d:.3f}, infl={v:.3f}, |S|={len(s)})"
                        for d, v, s in zip(seq_diams, seq_objs, seq_included)
                    )
                )
            )

            best_at_diameter = {}
            for diam, obj, incl in zip(seq_diams, seq_objs, seq_included):
                if diam not in best_at_diameter or obj > best_at_diameter[diam][0]:
                    best_at_diameter[diam] = (obj, incl)

            diameters = []
            best_objectives = []
            best_included_lists = []
            for diam in sorted(best_at_diameter.keys()):
                obj, incl = best_at_diameter[diam]
                diameters.append(diam)
                best_objectives.append(obj)
                best_included_lists.append(incl)

            return diameters, best_objectives, [-1 for _ in diameters], best_included_lists, runTime

        best_diams, best_objs, best_included_lists = [], [], []

        for target_diam in diameters:
            included = []
            remaining = set(range(n))
            best_obj = 0.0

            while remaining:
                best_idx = None
                best_score = -1
                best_obj_candidate = None
                curr_nodes = [self.nodes[i] for i in included]
                curr_obj = self.compute_influence(curr_nodes)

                for idx in remaining:
                    trial_included = included + [idx]
                    trial_diam = self._compute_diameter_from_indices(self.pairwise_costs, trial_included)
                    if trial_diam > target_diam:
                        continue

                    node_idx = self.nodes[idx]
                    obj = self.compute_influence(curr_nodes + [node_idx])
                    marginal_gain = obj - curr_obj

                    if included:
                        avg_dist = float(np.mean(self.pairwise_costs[idx, included]))
                    else:
                        avg_dist = 1.0

                    score = marginal_gain if avg_dist <= 0 else marginal_gain / avg_dist

                    if score > best_score:
                        best_score = score
                        best_idx = idx
                        best_obj_candidate = obj

                if best_idx is None:
                    break

                included.append(best_idx)
                remaining.remove(best_idx)
                best_obj = best_obj_candidate if best_obj_candidate is not None else best_obj

                if max_obj > 0 and best_obj >= 0.999 * max_obj:
                    break

            final_diam = self._compute_diameter_from_indices(self.pairwise_costs, included)
            best_diams.append(final_diam)
            best_objs.append(best_obj)
            best_included_lists.append(included.copy())

        runTime = time.perf_counter() - startTime
        best_at_diameter = {}
        for diam, obj, incl in zip(best_diams, best_objs, best_included_lists):
            if diam not in best_at_diameter or obj > best_at_diameter[diam][0]:
                best_at_diameter[diam] = (obj, incl)

        pruned_diams = []
        pruned_objs = []
        pruned_included = []
        for diam in sorted(best_at_diameter.keys()):
            obj, incl = best_at_diameter[diam]
            pruned_diams.append(diam)
            pruned_objs.append(obj)
            pruned_included.append(incl)

        logging.info(
            "PlainGreedyDistanceScaled finished: max_influence={:.3f}, runtime={:.3f}s".format(
                max(pruned_objs) if pruned_objs else 0.0, runTime
            )
        )
        logging.info(
            "PlainGreedyDistanceScaled solutions: {}".format(
                "; ".join(
                    f"(d={d:.3f}, infl={v:.3f}, |S|={len(s)})"
                    for d, v, s in zip(pruned_diams, pruned_objs, pruned_included)
                )
            )
        )

        return pruned_diams, pruned_objs, [-1 for _ in pruned_diams], pruned_included, runTime

    def topKDistanceScaled(self):
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

        for idx in ordered_indices:
            included.append(idx)
            curr_diam = self._compute_diameter_from_indices(self.pairwise_costs, included)
            curr_nodes = [self.nodes[i] for i in included]
            curr_obj = self.compute_influence(curr_nodes)
            seq_diams.append(curr_diam)
            seq_objs.append(curr_obj)
            seq_included.append(included.copy())

        runTime = time.perf_counter() - startTime
        best_at_diameter = {}
        for diam, obj, incl in zip(seq_diams, seq_objs, seq_included):
            if diam not in best_at_diameter or obj > best_at_diameter[diam][0]:
                best_at_diameter[diam] = (obj, incl)

        pruned_diams = []
        pruned_objs = []
        pruned_included = []
        for diam in sorted(best_at_diameter.keys()):
            obj, incl = best_at_diameter[diam]
            pruned_diams.append(diam)
            pruned_objs.append(obj)
            pruned_included.append(incl)

        logging.info(
            "TopKDistanceScaled finished: max_influence={:.3f}, runtime={:.3f}s".format(
                max(pruned_objs) if pruned_objs else 0.0, runTime
            )
        )
        logging.info(
            "TopKDistanceScaled solutions: {}".format(
                "; ".join(
                    f"(d={d:.3f}, infl={v:.3f}, |S|={len(s)})"
                    for d, v, s in zip(pruned_diams, pruned_objs, pruned_included)
                )
            )
        )

        return pruned_diams, pruned_objs, [-1 for _ in pruned_diams], pruned_included, runTime


def createGraph(data_path_file):
    """
    Create graph from influence dataset file, using largest connected components.
    """
    edges = []
    with open(data_path_file) as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                edge = [int(parts[0]), int(parts[1])]
                edges.append(edge)

    G = nx.DiGraph()
    for u, v in edges:
        G.add_edge(u, v)

    G_undir = G.to_undirected()

    if len(G_undir) > 0:
        components = [cc for cc in nx.connected_components(G_undir) if len(cc) <= 1000]
        if not components:
            components = list(nx.connected_components(G_undir))
        components_sorted = sorted(components, key=len, reverse=True)
        top_components = components_sorted[:10]
        selected_nodes = set().union(*top_components)
        G_undir = G_undir.subgraph(selected_nodes).copy()

    for u, v in G_undir.edges():
        G_undir[u][v]['weight'] = 0.1

    return G_undir


def import_influence_data(data_path):
    '''
    Import influence dataset
    '''
    G = createGraph(data_path)
    logging.info("Imported influence graph with {} nodes and {} edges".format(G.number_of_nodes(), G.number_of_edges()))
    return G


def compute_pairwise_costs_from_graph(G, nodes=None):
    '''
    Compute pairwise shortest-path distances for a graph using
    edge weights inversely proportional to shared neighbors.
    Returns (pairwise_costs, nodes).
    '''
    if nodes is None:
        nodes = list(G.nodes())
    nodes = list(nodes)
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    dist_mat = np.full((n, n), np.inf, dtype=float)

    G_weighted = G.copy()
    neighbor_sets = {node: set(G.neighbors(node)) for node in G.nodes()}
    for u, v in G_weighted.edges():
        shared = len(neighbor_sets[u] & neighbor_sets[v])
        G_weighted[u][v]['weight'] = float(np.exp(-0.1 * shared))

    for node in nodes:
        src_idx = node_index[node]
        dist_mat[src_idx, src_idx] = 0.0
        lengths = nx.single_source_dijkstra_path_length(G_weighted, node, weight='weight')
        for tgt, d in lengths.items():
            if tgt in node_index:
                dist_mat[src_idx, node_index[tgt]] = float(d)

    finite_vals = dist_mat[np.isfinite(dist_mat)]
    max_finite = float(np.max(finite_vals)) if finite_vals.size > 0 else 1.0
    dist_mat[~np.isfinite(dist_mat)] = 10.0 * max_finite

    return dist_mat, nodes


def sample_graph(G, sample_size=None, seed=None):
    '''
    Sample an induced subgraph of size sample_size if provided.
    '''
    nodes = list(G.nodes())
    n = len(nodes)
    if sample_size is None or sample_size <= 0 or sample_size >= n:
        return G, nodes
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n, size=sample_size, replace=False)
    chosen_nodes = [nodes[i] for i in chosen]
    subG = G.subgraph(chosen_nodes).copy()
    return subG, chosen_nodes
