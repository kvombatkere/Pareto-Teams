import time, pickle
from heapq import heappop, heappush, heapify
import numpy as np
import networkx as nx
from collections import defaultdict
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import logging

logging.basicConfig(format='%(asctime)s |%(levelname)s: %(message)s', level=logging.INFO)

class paretoCardinalityInfluence():
    '''
    Define a class for influence maximization with cardinality cost
    '''

    def __init__(self, G, k_max, num_samples=35, graph_samples=None):
        '''
        Initialize instance with graph and node costs
        ARGS:
            G           : networkx graph
            k_max       : cardinality constraint
            num_samples : number of graph samples for Monte Carlo
            graph_samples: pre-computed graph samples (optional)
        '''
        self.G = G
        self.k_max = k_max
        self.num_samples = num_samples
        self.nodes = list(G.nodes())
        self.n = len(self.nodes)
        self.kSolDict = {}
        self.reachable_nodes_memory = {}
        if graph_samples is not None:
            self.graph_samples = graph_samples
        else:
            self.initialize_graph_samples()
        logging.info("Initialized Pareto Influence - Cardinality Cost Instance, Num Nodes:{}, k={}".format(self.n, k_max))

    def initialize_graph_samples(self):
        '''
        Initialize Monte Carlo samples of the graph under independent cascade model
        '''
        self.graph_samples = []
        for i in range(self.num_samples):
            G_sample = nx.Graph()
            connected_components = defaultdict()
            for u, v, data in self.G.edges(data=True):
                success = np.random.uniform(0, 1)
                if success < data['weight']:
                    G_sample.add_edge(u, v)
            for c in nx.connected_components(G_sample):
                for node in c:
                    connected_components[node] = c
            self.graph_samples.append((G_sample, connected_components))

    def submodular_func_caching(self, solution_elements, item_id):
        """
        Submodular function without caching
        :param solution_elements: current solution nodes
        :param item_id: nodes to add
        :return: val, updated_solution_elements
        """
        all_nodes = solution_elements + item_id
        if not all_nodes:
            return 0, []

        spread = []
        for sample in self.graph_samples:
            connected_components = sample[2] if isinstance(sample, tuple) else sample
            active_nodes = set()
            for node in all_nodes:
                component = connected_components.get(node)
                if component is not None:
                    active_nodes.update(component)
            spread.append(len(active_nodes))

        val = np.mean(spread)
        return val, solution_elements + item_id

    def createNodeInfluenceMaxHeap(self):
        '''
        Initialize self.maxHeap with influence spreads for each node
        '''
        self.maxHeap = []
        heapify(self.maxHeap)
        
        for node in self.nodes:
            # Compute influence of single node
            influence = self.compute_influence([node])
            # push to maxheap - heapItem stored -influence, node
            heapItem = (influence * -1, node)
            heappush(self.maxHeap, heapItem)

        return 

    def compute_influence(self, nodes):
        '''
        Compute the expected influence spread of a set of nodes
        '''
        val, _ = self.submodular_func_caching(nodes, [])
        return val


    def greedyCardinality(self):
        '''
        Greedy Algorithm for Submodular Maximization under cardinality constraint
        '''
        startTime = time.perf_counter()

        solution_nodes = []
        curr_influence = 0
        k_val = 0

        # Create maxheap with influences
        self.createNodeInfluenceMaxHeap()

        # Assign nodes greedily using max heap until the solution has size k_max
        while len(self.maxHeap) > 0 and len(solution_nodes) < self.k_max:
            
            # Pop best node from maxHeap and compute marginal gain
            top_node_key = heappop(self.maxHeap)
            top_node = top_node_key[1]

            marginal_gain = self.compute_marginal_gain(solution_nodes, top_node)

            # Check node now on top - 2nd node on heap
            if len(self.maxHeap) > 0:
                second_node = self.maxHeap[0] 
                second_influence = second_node[0] * -1

                # If marginal gain of top node is better we add to solution
                if marginal_gain >= second_influence:
                    k_val += 1
                    solution_nodes.append(top_node)
                    curr_influence += marginal_gain
                    self.kSolDict[k_val] = {"Nodes": solution_nodes.copy(), "Influence": curr_influence}
                    logging.debug("k = {}, Adding node {}, curr_influence={:.3f}".format(k_val, top_node, curr_influence))
            
                # Otherwise re-insert top node into heap with updated marginal gain
                else:
                    updated_top_node = (marginal_gain * -1, top_node)
                    heappush(self.maxHeap, updated_top_node)
            else:
                # Last node
                if marginal_gain > 0:
                    k_val += 1
                    solution_nodes.append(top_node)
                    curr_influence += marginal_gain
                    self.kSolDict[k_val] = {"Nodes": solution_nodes.copy(), "Influence": curr_influence}

        runTime = time.perf_counter() - startTime
        logging.info("Cardinality Greedy Solution for k_max:{}, Influence:{:.3f}, Runtime = {:.2f} seconds".format(len(solution_nodes), curr_influence, runTime))

        return solution_nodes, curr_influence, runTime
    
    def compute_marginal_gain(self, current_nodes, new_node):
        '''
        Compute marginal gain of adding new_node to current_nodes
        '''
        if not self.graph_samples:
            return 0

        marginal_gain = 0
        for sample in self.graph_samples:
            connected_components = sample[2] if isinstance(sample, tuple) else sample
            active_nodes = set()
            for node in current_nodes:
                component = connected_components.get(node)
                if component is not None:
                    active_nodes.update(component)
            component_new = connected_components.get(new_node)
            if component_new is not None:
                marginal_gain += len(component_new - active_nodes)

        return marginal_gain / len(self.graph_samples)

    def top_k(self):
        '''
        Top-k Algorithm: Select the top k nodes from the heap without updates.
        Influences are computed w.r.t. the empty set (i.e., individual influences).
        '''
        startTime = time.perf_counter()

        solution_nodes = []
        curr_influence = 0

        # Create maxheap with influences
        self.createNodeInfluenceMaxHeap()

        # Select top k nodes
        k_val = 0
        while self.maxHeap and k_val < self.k_max:
            top_node_key = heappop(self.maxHeap)
            top_node = top_node_key[1]

            marginal_gain = self.compute_marginal_gain(solution_nodes, top_node)
            if marginal_gain > 0:
                k_val += 1
                solution_nodes.append(top_node)
                curr_influence += marginal_gain
                self.kSolDict[k_val] = {"Nodes": solution_nodes.copy(), "Influence": curr_influence}
                logging.debug("k = {}, Adding top node {}, curr_influence={:.3f}".format(k_val, top_node, curr_influence))

        runTime = time.perf_counter() - startTime
        logging.info("Top-k Solution for k_max:{}, Influence:{:.3f}, Runtime = {:.2f} seconds".format(len(solution_nodes), curr_influence, runTime))

        return solution_nodes, curr_influence, runTime

    def random_selection(self):
        '''
        Random Algorithm: Randomly select k distinct nodes.
        '''
        startTime = time.perf_counter()

        # Randomly select k_max distinct node indices
        selected_indices = np.random.choice(self.n, size=self.k_max, replace=False)

        solution_nodes = [self.nodes[i] for i in selected_indices]

        curr_influence = self.compute_influence(solution_nodes)

        # Populate kSolDict for consistency
        for k_val in range(1, self.k_max + 1):
            partial_nodes = solution_nodes[:k_val]
            partial_influence = self.compute_influence(partial_nodes)
            self.kSolDict[k_val] = {"Nodes": partial_nodes, "Influence": partial_influence}

        runTime = time.perf_counter() - startTime
        logging.info("Random Selection Solution for k_max:{}, Influence:{:.3f}, Runtime = {:.2f} seconds".format(len(solution_nodes), curr_influence, runTime))

        return solution_nodes, curr_influence, runTime

def createGraph(data_path_file):
    """
    Create graph from influence dataset file, using the two largest connected components
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
    for edge in edges:
        u, v = edge[0], edge[1]
        G.add_edge(u, v)

    # Convert to undirected for influence spread (assuming undirected propagation)
    G_undir = G.to_undirected()

    # Take the two largest connected components (prefer components with size <= 1000)
    if len(G_undir) > 0:
        components = [cc for cc in nx.connected_components(G_undir) if len(cc) <= 1000]
        if not components:
            components = list(nx.connected_components(G_undir))

        components_sorted = sorted(components, key=len, reverse=True)
        top_components = components_sorted[:10]
        selected_nodes = set().union(*top_components)
        G_undir = G_undir.subgraph(selected_nodes).copy()  # Create a copy of the subgraph

    # Add default weights (can be adjusted)
    for u, v in G_undir.edges():
        G_undir[u][v]['weight'] = 0.1  # placeholder probability

    return G_undir

def import_influence_data(data_path):
    '''
    Import influence dataset
    '''
    G = createGraph(data_path)

    logging.info("Imported influence graph with {} nodes and {} edges".format(G.number_of_nodes(), G.number_of_edges()))

    return G