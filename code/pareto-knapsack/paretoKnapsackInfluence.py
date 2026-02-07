import time, pickle
from heapq import heappop, heappush, heapify
import numpy as np
import networkx as nx
from collections import defaultdict
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import logging

logging.basicConfig(format='%(asctime)s |%(levelname)s: %(message)s', level=logging.INFO)

class paretoKnapsackInfluence():
    '''
    Define a class for influence maximization with knapsack cost
    '''

    def __init__(self, G, node_costs, budget, num_samples=35, graph_samples=None):
        '''
        Initialize instance with graph and node costs
        ARGS:
            G           : networkx graph
            node_costs  : dict of node costs
            budget      : knapsack budget
            num_samples : number of graph samples for Monte Carlo
            graph_samples: pre-computed graph samples (optional)
        '''
        self.G = G
        self.node_costs = node_costs
        self.B = budget
        self.num_samples = num_samples
        self.nodes = list(G.nodes())
        self.n = len(self.nodes)
        if graph_samples is not None:
            self.graph_samples = graph_samples
        else:
            self.initialize_graph_samples()
        
        logging.info("Initialized Pareto Influence - Knapsack Cost Instance, Num Nodes:{}, Budget={}".format(self.n, self.B))


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
            self.graph_samples.append(connected_components)

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
            
        # logging.info("Computed spread for solution {}, added item {}, spread_arr={}".format(solution_elements, item_id, spread))
        val = np.mean(spread)
        return val, solution_elements + item_id

    def compute_influence(self, nodes):
        '''
        Compute the expected influence spread of a set of nodes
        '''
        val, _ = self.submodular_func_caching(nodes, [])
        return val

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

    def getNodeInfluenceAdd(self, influence_x, node_index, curr_solution, curr_influence):
        '''
        Helper function to get influence addition of new node
        '''
        marginal_gain = self.compute_marginal_gain(curr_solution, self.nodes[node_index])
        influence_add = curr_influence + marginal_gain
        influence_ratio_add = (min(influence_x, influence_add) - curr_influence) / self.node_costs[self.nodes[node_index]]
        return influence_ratio_add
    

    def createNodeInfluenceMaxHeap(self):
        '''
        Initialize self.maxHeap with node influence gains for each node
        '''
        #Create max heap to store node influence gains
        self.maxHeap = []
        heapify(self.maxHeap)

        for i, node in enumerate(self.nodes):
            #Compute node influence gain
            node_influence = self.compute_influence([node])
            node_weight = node_influence / self.node_costs[node]

            #push to maxheap - heapItem stored -gain, node index and cost
            heapItem = (node_weight*-1, i, self.node_costs[node])
            heappush(self.maxHeap, heapItem)

        return 
    

    def plainGreedy(self):
        '''
        Adapt Plain Greedy Algorithm from  Feldman, Nutov, Shoham 2021; Practical Budgeted Submodular Maximization
        Run with input nodes instead of sets
        '''
        startTime = time.perf_counter()

        #Solution nodes
        solution_nodes = []

        curr_influence, curr_cost = 0, 0

        #Create maxheap with influence gains
        self.createNodeInfluenceMaxHeap()

        #Assign nodes greedily using max heap
        #Check if there is a node with cost that fits in budget
        while len(self.maxHeap) > 1 and (min(key[2] for key in self.maxHeap) <= (self.B - curr_cost)):

            #Pop best node from maxHeap and compute marginal gain
            top_node_key = heappop(self.maxHeap)
            top_node_indx, top_node_cost = top_node_key[1], top_node_key[2]
            top_node = self.nodes[top_node_indx]

            influence_with_top_node = self.compute_influence(solution_nodes + [top_node])
            top_node_marginal_gain = (influence_with_top_node - curr_influence) / top_node_cost

            #Check node now on top - 2nd node on heap
            second_node = self.maxHeap[0]
            second_node_heap_gain = second_node[0] * -1

            #If marginal gain of top node is better we add to solution
            if top_node_marginal_gain >= second_node_heap_gain:
                #Only add if node is within budget
                if top_node_cost + curr_cost <= self.B:
                    solution_nodes.append(top_node)
                    curr_influence = influence_with_top_node
                    curr_cost += top_node_cost
                    logging.debug("Adding node {}, curr_influence={:.3f}, curr_cost={}".format(top_node, curr_influence, curr_cost))

            #Otherwise re-insert top node into heap with updated marginal gain
            else:
                updated_top_node = (top_node_marginal_gain*-1, top_node_indx, top_node_cost)
                heappush(self.maxHeap, updated_top_node)

        runTime = time.perf_counter() - startTime
        logging.debug("Plain Greedy Solution:{}, Influence:{:.3f}, Cost:{}, Runtime = {:.2f} seconds".format(solution_nodes, curr_influence, curr_cost, runTime))

        return solution_nodes, curr_influence, curr_cost, runTime
    

    def greedyPlus(self):
        '''
        Greedy Plus Algorithm from  Feldman, Nutov, Shoham 2021; Practical Budgeted Submodular Maximization
        Greedy returns the better solution among the output of Plain Greedy and the best feasible solution
        that can be obtained by combining any solution that Plain Greedy had at some iteration
        with a single node.
        '''
        startTime = time.perf_counter()

        #Get plain greedy solution
        sol_nodes, best_influence, best_cost, pg_runtime = self.plainGreedy()

        logging.debug("=="*50)
        best_nodes_list = []

        #Loop over solution in each iteration of plain greedy
        for i, node_i in enumerate(sol_nodes):
            feasible_nodes = sol_nodes[:i+1]
            feasible_cost = sum(self.node_costs[node] for node in feasible_nodes)
            logging.debug("Trying incremental solution:{}, cost:{}".format(feasible_nodes, feasible_cost))

            for j, node_j in enumerate(self.nodes):
                #If adding a single node doesn't violate budget
                if node_j not in feasible_nodes and feasible_cost + self.node_costs[node_j] <= self.B:
                    #Compute influence by adding node to incremental solution
                    added_influence = self.compute_influence(feasible_nodes + [node_j])

                    #If this solution is better than original solution, store it
                    if added_influence > best_influence:
                        best_nodes_list = feasible_nodes + [node_j]
                        best_influence = added_influence
                        best_cost = feasible_cost + self.node_costs[node_j]
                        logging.debug("New feasible solution yielded better influence! {}, influence={:.3f}, cost={}".format(best_nodes_list, best_influence, best_cost))

        #Return original solution if that is better
        if len(best_nodes_list) == 0:
            logging.debug("Original Plain Greedy Solution was best!")
            best_nodes_list = sol_nodes

        runTime = time.perf_counter() - startTime
        logging.debug("Greedy+ Solution:{}, Influence:{:.3f}, Cost:{}, Runtime = {:.2f} seconds".format(best_nodes_list, best_influence, best_cost, runTime))

        #Return solution
        return best_nodes_list, best_influence, best_cost, runTime


    def top_k(self):
        '''
        Budget-threshold heuristic: select nodes by highest cost-scaled marginal gain
        with respect to the empty set (i.e., influence of single node / cost),
        adding nodes until the budget is exhausted.
        Only considers nodes that are individually within the budget.
        '''
        startTime = time.perf_counter()

        node_scores = []
        for i, node in enumerate(self.nodes):
            cost = self.node_costs[node]
            if cost <= self.B and cost > 0:
                node_influence = self.compute_influence([node])
                node_scores.append((node_influence / cost, i))

        node_scores.sort(key=lambda x: x[0], reverse=True)
        selected_indices = []
        curr_cost = 0.0
        for _, idx in node_scores:
            node = self.nodes[idx]
            cost = self.node_costs[node]
            if curr_cost + cost <= self.B:
                selected_indices.append(idx)
                curr_cost += cost

        solution_nodes = [self.nodes[idx] for idx in selected_indices]
        curr_influence = self.compute_influence(solution_nodes) if solution_nodes else 0

        runTime = time.perf_counter() - startTime
        logging.debug("Top-k (cost-scaled, budget-feasible) Solution, Influence:{:.3f}, Cost:{}, Runtime = {:.2f} seconds".format(curr_influence, curr_cost, runTime))

        return solution_nodes, curr_influence, curr_cost, runTime
    

    def createmaxHeap2Guess(self, node_pair_key, node_pair_data):
        '''
        Initialize self.maxHeap2Guess with node influence gains for each node that is not in the pair
        '''
        #Create max heap to store influence gains with respect to new objective function
        self.maxHeap2Guess = []
        heapify(self.maxHeap2Guess)

        #Compute influence and cost of pair
        nodePairInfluence = self.compute_influence(node_pair_data[0])
        nodePairCost = node_pair_data[1]

        for i, node_i in enumerate(self.nodes):
            if node_i not in node_pair_data[0] and (self.node_costs[node_i] + nodePairCost <= self.B): #Only add new nodes that fit budget
                #Compute marginal influence of new node
                marginal_influence = self.compute_marginal_gain(node_pair_data[0], node_i)
                node_weight = marginal_influence / self.node_costs[node_i]

                #push to maxheap - heapItem stored -gain, node index and cost
                heapItem = (node_weight*-1, i, self.node_costs[node_i])
                heappush(self.maxHeap2Guess, heapItem)

        return nodePairInfluence, nodePairCost
    
    def twoGuessPlainGreedy(self):
        '''
        2-Guess Plain Greedy from  Feldman, Nutov, Shoham 2021; Practical Budgeted Submodular Maximization
        '''
        startTime = time.perf_counter()

        allNodePairs = {}
        #Get node pairs and store list of nodes and costs
        for i, node_i in enumerate(self.nodes):
            for j, node_j in enumerate(self.nodes):
                if i < j:
                    node_pair_key = (i, j)
                    node_pair_nodes = [node_i, node_j]
                    node_pair_cost = self.node_costs[node_i] + self.node_costs[node_j]

                    #Only add nodes who cost less than the budget
                    if node_pair_cost <= self.B:
                        allNodePairs[node_pair_key] = [node_pair_nodes, node_pair_cost]

        logging.debug("Created allNodePairs with {} pairs".format(len(allNodePairs)))

        #Get best single node solution
        best_single_node, best_single_infl, best_single_cost = [], 0, 0
        for i, node_i in enumerate(self.nodes):
            if self.node_costs[node_i] <= self.B:
                node_i_infl = self.compute_influence([node_i])

                if node_i_infl > best_single_infl:
                    best_single_infl = node_i_infl
                    best_single_cost = self.node_costs[node_i]
                    best_single_node = [node_i]

        #Keep track of all solutions and their costs
        solutionDict = {}
        best_sol_nodes, best_influence, best_cost = [], 0, 0

        #Run Plain Greedy for each pair
        for pair_key, pair_data in allNodePairs.items():

            #Create priority queue with all other nodes for this run
            #Initialize variables for this greedy run
            curr_influence, curr_cost = self.createmaxHeap2Guess(node_pair_key=pair_key, node_pair_data=pair_data)
            solution_nodes = pair_data[0].copy()

            #Assign nodes greedily using maxHeap2Guess
            #Check if there is a node with cost that fits in budget
            while len(self.maxHeap2Guess) > 1 and (min(key[2] for key in self.maxHeap2Guess) <= (self.B - curr_cost)):

                #Pop best node from maxHeap2Guess and compute marginal gain
                top_node_key = heappop(self.maxHeap2Guess)
                top_node_indx, top_node_cost = top_node_key[1], top_node_key[2]
                top_node = self.nodes[top_node_indx]

                influence_with_top_node = self.compute_influence(solution_nodes + [top_node])
                top_node_marginal_gain = (influence_with_top_node - curr_influence) / top_node_cost

                #Check node now on top - 2nd node on heap
                second_node = self.maxHeap2Guess[0]
                second_node_heap_gain = second_node[0] * -1

                #If marginal gain of top node is better we add to solution
                if top_node_marginal_gain >= second_node_heap_gain:
                    #Only add if node is within budget
                    if top_node_cost + curr_cost <= self.B:
                        solution_nodes.append(top_node)
                        curr_influence = influence_with_top_node
                        curr_cost += top_node_cost
                        logging.debug("Adding node {}, curr_influence={:.3f}, curr_cost={}".format(top_node, curr_influence, curr_cost))

                #Otherwise re-insert top node into heap with updated marginal gain
                else:
                    updated_top_node = (top_node_marginal_gain*-1, top_node_indx, top_node_cost)
                    heappush(self.maxHeap2Guess, updated_top_node)

            #Add solution to dict
            logging.debug("Computed Pair Solution for seed{}, nodes:{}, influence={:.3f}, cost={}".format(pair_key, solution_nodes, curr_influence, curr_cost))
            solutionDict[pair_key] = {'nodes':solution_nodes, 'influence':curr_influence, 'cost':curr_cost}
            if curr_influence > best_influence:
                best_influence = curr_influence
                best_cost = curr_cost
                best_sol_nodes = solution_nodes

        #Compare with best single node solution - if they are equivalent choose single
        if best_single_infl >= best_influence:
            best_influence = best_single_infl
            best_cost = best_single_cost
            best_sol_nodes = best_single_node

        runTime = time.perf_counter() - startTime
        logging.debug("2-Guess Plain Greedy Solution:{}, Influence:{:.3f}, Cost:{}, Runtime = {:.2f} seconds".format(best_sol_nodes, best_influence, best_cost, runTime))

        return best_sol_nodes, best_influence, best_cost, runTime
    
    def prefixParetoGreedy_2Guess(self):
        '''
        Prefix Pareto Greedy Algorithm - implemented as a variant of 2-Guess Plain Greedy
        Tracks Pareto optimal influence-cost tradeoffs
        '''
        startTime = time.perf_counter()

        #Dictionary to track best influence spread for each cost value
        cost_influence_map = {}
        allNodePairs = {}

        #Get node pairs and store influence and costs
        for i, node_i in enumerate(self.nodes):
            for j, node_j in enumerate(self.nodes):
                if i < j:
                    node_pair_key = (i, j)
                    node_pair_nodes = [node_i, node_j]
                    node_pair_cost = self.node_costs[node_i] + self.node_costs[node_j]

                    #Only add nodes who cost less than the budget
                    if node_pair_cost <= self.B:
                        allNodePairs[node_pair_key] = [node_pair_nodes, node_pair_cost]

        logging.debug("Created allNodePairs with {} pairs".format(len(allNodePairs)))

        #Track single node influence solutions
        for i, node_i in enumerate(self.nodes):
            if self.node_costs[node_i] <= self.B:
                node_i_infl = self.compute_influence([node_i])
                #Update influence map with single node solution
                if self.node_costs[node_i] not in cost_influence_map or node_i_infl > cost_influence_map[self.node_costs[node_i]][0]:
                    cost_influence_map[self.node_costs[node_i]] = [node_i_infl, [node_i]]

        #Run Greedy for each pair and track prefixes
        for pair_key, pair_data in allNodePairs.items():
            
            #Create priority queue with all other nodes for this run
            #Initialize variables for this greedy run
            curr_influence, curr_cost = self.createmaxHeap2Guess(node_pair_key=pair_key, node_pair_data=pair_data)
            solution_nodes = pair_data[0].copy()
            
            #Track initial pair influence
            if curr_cost not in cost_influence_map or curr_influence > cost_influence_map[curr_cost][0]:
                cost_influence_map[curr_cost] = [curr_influence, solution_nodes.copy()]

            #Assign nodes greedily using maxHeap2Guess
            #Check if there is a node with cost that fits in budget
            while len(self.maxHeap2Guess) > 1 and (min(key[2] for key in self.maxHeap2Guess) <= (self.B - curr_cost)):
                
                #Pop best node from maxHeap2Guess and compute marginal gain
                top_node_key = heappop(self.maxHeap2Guess)
                top_node_indx, top_node_cost = top_node_key[1], top_node_key[2]
                top_node = self.nodes[top_node_indx]

                influence_with_top_node = self.compute_influence(solution_nodes + [top_node])
                top_node_marginal_gain = (influence_with_top_node - curr_influence) / top_node_cost

                #Check node now on top - 2nd node on heap
                second_node = self.maxHeap2Guess[0] 
                second_node_heap_gain = second_node[0]*-1

                #If marginal gain of top node is better we add to solution
                if top_node_marginal_gain >= second_node_heap_gain:
                    #Only add if node is within budget
                    if top_node_cost + curr_cost <= self.B:
                        solution_nodes.append(top_node)
                        curr_influence = influence_with_top_node
                        curr_cost += top_node_cost

                        #Track best influence for this cost
                        if curr_cost not in cost_influence_map or curr_influence > cost_influence_map[curr_cost][0]:
                            cost_influence_map[curr_cost] = [curr_influence, solution_nodes.copy()]
                        logging.debug("Adding node {}, curr_influence={:.3f}, curr_cost={}".format(top_node, curr_influence, curr_cost))
                
                #Otherwise re-insert top node into heap with updated marginal gain
                else:
                    updated_top_node = (top_node_marginal_gain*-1, top_node_indx, top_node_cost)
                    heappush(self.maxHeap2Guess, updated_top_node)

        #Prune to only keep Pareto optimal influence-cost tradeoffs
        prunedBudgets, prunedInfluences = [], []
        currentInfl = 0
        for b_prime in sorted(cost_influence_map.keys()):
            if cost_influence_map[b_prime][0] > currentInfl:
                currentInfl = cost_influence_map[b_prime][0]
                prunedBudgets.append(b_prime)
                prunedInfluences.append(currentInfl)
                logging.debug("Pareto Optimal - Cost: {}, Influence: {:.3f}, Nodes: {}".format(b_prime, cost_influence_map[b_prime][0], cost_influence_map[b_prime][1]))

        runTime = time.perf_counter() - startTime
        logging.debug("Prefix Pareto Greedy Runtime = {:.2f} seconds".format(runTime))

        return prunedBudgets, prunedInfluences, cost_influence_map, runTime
    

    def createmaxHeap1Guess(self, seed_node, seed_node_cost, seed_node_index):
        '''
        Initialize self.maxHeap1Guess with node influence gains for each node that is not the seed
        '''
        #Create max heap to store influence gains with respect to new objective function
        self.maxHeap1Guess = []
        heapify(self.maxHeap1Guess)

        #Compute influence and cost of seed
        seed_influence = self.compute_influence([seed_node])

        for i, node_i in enumerate(self.nodes):
            if i != seed_node_index and (self.node_costs[node_i] + seed_node_cost <= self.B): #Only add new nodes that fit budget
                #Compute marginal influence of new node
                marginal_influence = self.compute_marginal_gain([seed_node], node_i)
                node_weight = marginal_influence / self.node_costs[node_i]

                #push to maxheap - heapItem stored -gain, node index and cost
                heapItem = (node_weight*-1, i, self.node_costs[node_i])
                heappush(self.maxHeap1Guess, heapItem)

        return seed_influence, seed_node_cost
        

    def oneGuessGreedyPlus(self):
        '''
        1-Guess Greedy+ from Feldman, Nutov, Shoham 2021; Practical Budgeted Submodular Maximization
        '''
        startTime = time.perf_counter()

        #Keep track of all solutions and their costs
        solutionDict = {}
        best_sol_nodes, best_influence, best_cost = [], 0, 0

        #Iterate over all single node seeds
        for i, node_i in enumerate(self.nodes):
            if self.node_costs[node_i] <= self.B:
                node_i_infl = self.compute_influence([node_i])

                #Create priority queue with all other nodes for this run
                #Initialize variables for this greedy run
                curr_influence, curr_cost = self.createmaxHeap1Guess(seed_node=node_i,
                                                                    seed_node_cost=self.node_costs[node_i],
                                                                    seed_node_index=i)

                solution_nodes = [node_i]

                #Assign nodes greedily using max heap
                #Check if there is a node with cost that fits in budget
                while len(self.maxHeap1Guess) > 1 and (min(key[2] for key in self.maxHeap1Guess) <= (self.B - curr_cost)):

                    #Pop best node from maxHeap1Guess and compute marginal gain
                    top_node_key = heappop(self.maxHeap1Guess)
                    top_node_indx, top_node_cost = top_node_key[1], top_node_key[2]
                    top_node = self.nodes[top_node_indx]

                    influence_with_top_node = self.compute_influence(solution_nodes + [top_node])
                    top_node_marginal_gain = (influence_with_top_node - curr_influence) / top_node_cost

                    #Check node now on top - 2nd node on heap
                    second_node = self.maxHeap1Guess[0]
                    second_node_heap_gain = second_node[0] * -1

                    #If marginal gain of top node is better we add to solution
                    if top_node_marginal_gain >= second_node_heap_gain:
                        #Only add if node is within budget
                        if top_node_cost + curr_cost <= self.B:
                            solution_nodes.append(top_node)
                            curr_influence = influence_with_top_node
                            curr_cost += top_node_cost
                            logging.debug("Adding node {}, curr_influence={:.3f}, curr_cost={}".format(top_node, curr_influence, curr_cost))

                    #Otherwise re-insert top node into heap with updated marginal gain
                    else:
                        updated_top_node = (top_node_marginal_gain*-1, top_node_indx, top_node_cost)
                        heappush(self.maxHeap1Guess, updated_top_node)

                #Store results for run with seed i
                seed_i_influence, seed_i_cost = curr_influence, curr_cost
                seed_i_nodes = solution_nodes.copy()
                # #Perform Greedy+ check - Loop over solution in each iteration of plain greedy
                # for j, node_j in enumerate(solution_nodes):
                #     feasible_nodes = solution_nodes[:j+1]
                #     feasible_cost = sum(self.node_costs[node] for node in feasible_nodes)
                #     logging.debug("Trying incremental solution:{}, cost:{}".format(feasible_nodes, feasible_cost))

                #     for k, node_k in enumerate(self.nodes):
                #         #If adding a single node doesn't violate budget
                #         if node_k not in feasible_nodes and feasible_cost + self.node_costs[node_k] <= self.B:
                #             #Compute influence by adding node to incremental solution
                #             added_influence = self.compute_influence(feasible_nodes + [node_k])

                #             #If this solution is better than original solution, store it
                #             if added_influence > seed_i_influence:
                #                 seed_i_nodes = feasible_nodes + [node_k]
                #                 seed_i_influence = added_influence
                #                 seed_i_cost = feasible_cost + self.node_costs[node_k]
                #                 logging.debug("New feasible seed solution yielded better influence! {}, influence={:.3f}, cost={}".format(seed_i_nodes,
                #                                                                                                                        seed_i_influence, seed_i_cost))

                #Store best solution for seed i
                logging.debug("Best solution for seed {}, nodes:{}, influence={:.3f}, cost={}".format(i, seed_i_nodes, seed_i_influence, seed_i_cost))
                solutionDict[i] = {'nodes':seed_i_nodes, 'influence':seed_i_influence, 'cost':seed_i_cost}
                #Keep track of best solution across all seeds
                if seed_i_influence > best_influence:
                    best_influence = seed_i_influence
                    best_cost = seed_i_cost
                    best_sol_nodes = seed_i_nodes

        runTime = time.perf_counter() - startTime
        logging.debug("1-Guess Greedy+ Solution:{}, Influence:{:.3f}, Cost:{}, Runtime = {:.2f} seconds".format(best_sol_nodes, best_influence, best_cost, runTime))

        return best_sol_nodes, best_influence, best_cost, runTime


    def prefixParetoGreedy_1Guess(self):
        '''
        Prefix Pareto Greedy Algorithm - implemented as a variant of 1-Guess Greedy
        Tracks Pareto optimal influence-cost tradeoffs
        '''
        startTime = time.perf_counter()

        #Dictionary to track best influence spread for each cost value
        cost_influence_map = {}

        #Iterate over all single node seeds
        for i, node_i in enumerate(self.nodes):
            if self.node_costs[node_i] <= self.B:
                node_i_infl = self.compute_influence([node_i])

                #Track influence of single node solution
                if self.node_costs[node_i] not in cost_influence_map or node_i_infl > cost_influence_map[self.node_costs[node_i]][0]:
                    cost_influence_map[self.node_costs[node_i]] = [node_i_infl, [node_i]]

                #Create priority queue with all other nodes for this run
                #Initialize variables for this greedy run
                curr_influence, curr_cost = self.createmaxHeap1Guess(seed_node=node_i, seed_node_cost=self.node_costs[node_i], 
                                                                    seed_node_index=i)
                solution_nodes = [node_i]

                #Assign nodes greedily using max heap
                #Check if there is a node with cost that fits in budget
                while len(self.maxHeap1Guess) > 1 and (min(key[2] for key in self.maxHeap1Guess) <= (self.B - curr_cost)):
                    
                    #Pop best node from maxHeap1Guess and compute marginal gain
                    top_node_key = heappop(self.maxHeap1Guess)
                    top_node_indx, top_node_cost = top_node_key[1], top_node_key[2]
                    top_node = self.nodes[top_node_indx]

                    influence_with_top_node = self.compute_influence(solution_nodes + [top_node])
                    top_node_marginal_gain = (influence_with_top_node - curr_influence) / top_node_cost

                    #Check node now on top - 2nd node on heap
                    second_node = self.maxHeap1Guess[0] 
                    second_node_heap_gain = second_node[0]*-1

                    #If marginal gain of top node is better we add to solution
                    if top_node_marginal_gain >= second_node_heap_gain:
                        #Only add if node is within budget
                        if top_node_cost + curr_cost <= self.B:
                            solution_nodes.append(top_node)
                            curr_influence = influence_with_top_node
                            curr_cost += top_node_cost

                            #Track best influence for this cost
                            if curr_cost not in cost_influence_map or curr_influence > cost_influence_map[curr_cost][0]:
                                cost_influence_map[curr_cost] = [curr_influence, solution_nodes.copy()]
                            logging.debug("Adding node {}, curr_influence={:.3f}, curr_cost={}".format(top_node, curr_influence, curr_cost))
                    
                    #Otherwise re-insert top node into heap with updated marginal gain
                    else:
                        updated_top_node = (top_node_marginal_gain*-1, top_node_indx, top_node_cost)
                        heappush(self.maxHeap1Guess, updated_top_node)

        #Prune to only keep Pareto optimal influence-cost tradeoffs
        prunedBudgets, prunedInfluences = [], []
        currentInfl = 0
        for b_prime in sorted(cost_influence_map.keys()):
            if cost_influence_map[b_prime][0] > currentInfl:
                currentInfl = cost_influence_map[b_prime][0]
                prunedBudgets.append(b_prime)
                prunedInfluences.append(currentInfl)
                logging.debug("Pareto Optimal - Cost: {}, Influence: {:.3f}, Nodes: {}".format(b_prime, cost_influence_map[b_prime][0], cost_influence_map[b_prime][1]))

        runTime = time.perf_counter() - startTime
        logging.debug("Prefix Pareto Greedy - 1 Guess Runtime = {:.2f} seconds".format(runTime))

        return prunedBudgets, prunedInfluences, cost_influence_map, runTime


    def F_Greedy(self):
        '''
        Linear influence sweep: for each discrete influence level, find a minimum-cost
        solution using weighted greedy (marginal gain scaled by cost) with seed size 1.
        Then prune dominated solutions.
        '''
        startTime = time.perf_counter()

        if self.n == 0:
            return [], [], {}, 0.0

        # Maximum achievable influence (using all nodes)
        max_influence = self.compute_influence(self.nodes)
        if max_influence <= 0:
            return [], [], {}, 0.0

        # Discrete influence targets: 1, 2, ..., ceil(max_influence)
        max_target = int(np.ceil(max_influence))
        target_influences = list(range(1, max_target + 1))

        # Track best solution per target influence
        cost_influence_map = {}

        for infl_x in target_influences:
            # Try all single-node seeds for this target influence
            for i, node_i in enumerate(self.nodes):
                if self.node_costs[node_i] > self.B:
                    continue

                curr_influence, curr_cost = self.createmaxHeap1Guess(
                    seed_node=node_i,
                    seed_node_cost=self.node_costs[node_i],
                    seed_node_index=i
                )
                curr_solution_nodes = [node_i]

                # Weighted greedy until reaching target influence or no feasible node
                while len(self.maxHeap1Guess) > 1 and (min(key[2] for key in self.maxHeap1Guess) <= (self.B - curr_cost)) and (curr_influence < infl_x):
                    top_node_key = heappop(self.maxHeap1Guess)
                    top_node_indx, top_node_cost = top_node_key[1], top_node_key[2]
                    top_node = self.nodes[top_node_indx]

                    influence_with_top_node = self.compute_influence(curr_solution_nodes + [top_node])
                    top_node_marginal_gain = (influence_with_top_node - curr_influence) / top_node_cost

                    # Compare against next best heap gain
                    second_node = self.maxHeap1Guess[0]
                    second_node_heap_gain = second_node[0] * -1

                    if top_node_marginal_gain >= second_node_heap_gain:
                        if top_node_cost + curr_cost <= self.B:
                            curr_solution_nodes.append(top_node)
                            curr_influence = influence_with_top_node
                            curr_cost += top_node_cost
                            logging.debug("Adding node {}, curr_influence={:.3f}, curr_cost={}".format(top_node, curr_influence, curr_cost))
                    else:
                        updated_top_node = (top_node_marginal_gain * -1, top_node_indx, top_node_cost)
                        heappush(self.maxHeap1Guess, updated_top_node)

                # Store if target met within budget
                if curr_influence >= infl_x:
                    if infl_x not in cost_influence_map or curr_cost < cost_influence_map[infl_x][0]:
                        cost_influence_map[infl_x] = [curr_cost, curr_solution_nodes.copy()]

        # Prune dominated solutions: keep strictly increasing influence as cost increases
        prunedBudgets, prunedInfluences = [], []
        pairs = [(data[0], infl) for infl, data in cost_influence_map.items()]
        pairs.sort(key=lambda x: x[0])  # sort by cost
        best_infl = -1.0
        for cost, infl in pairs:
            if infl > best_infl:
                best_infl = infl
                prunedBudgets.append(cost)
                prunedInfluences.append(infl)
                logging.debug("Approx. Pareto Influence: {}, Cost: {}, Nodes: {}".format(infl, cost, cost_influence_map[infl][1]))

        runTime = time.perf_counter() - startTime
        logging.debug("Coverage Linear Runtime = {:.2f} seconds".format(runTime))

        return prunedBudgets, prunedInfluences, cost_influence_map, runTime


def createGraph(data_path_file):
    '''
    Create graph from influence dataset file
    '''
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

    # Take only the largest connected component with size <= 500
    if len(G_undir) > 0:
        components = [cc for cc in nx.connected_components(G_undir) if len(cc) <= 500]
        if components:
            largest_cc = max(components, key=len)
        else:
            # If no component <=500, take the largest overall
            largest_cc = max(nx.connected_components(G_undir), key=len)
        G_undir = G_undir.subgraph(largest_cc).copy()  # Create a copy of the subgraph

    # Add default weights (can be adjusted)
    for u, v in G_undir.edges():
        G_undir[u][v]['weight'] = 0.1  # placeholder probability

    neighbors = defaultdict(list)
    for u, v in G_undir.edges():
        neighbors[u].append(v)
        neighbors[v].append(u)
    
    return G_undir, neighbors

def import_influence_data(data_path, node_costs=None):
    '''
    Import influence dataset
    '''
    G, neighbors = createGraph(data_path)
    
    if node_costs is None:
        # Default costs: node degree in the influence graph
        node_costs = {node: G.degree(node) for node in G.nodes()}
    
    logging.info("Imported influence graph with {} nodes and {} edges".format(G.number_of_nodes(), G.number_of_edges()))
    
    return G, node_costs