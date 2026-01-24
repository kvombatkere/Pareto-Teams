import time, pickle
from heapq import heappop, heappush, heapify
import numpy as np
import networkx as nx
from collections import defaultdict
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
        self.reachable_nodes_memory = {}
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
            neighbors = defaultdict(set)
            connected_components = defaultdict()
            for u, v, data in self.G.edges(data=True):
                success = np.random.uniform(0, 1)
                if success < data['weight']:
                    G_sample.add_edge(u, v)
                    neighbors[u].add(v)
                    neighbors[v].add(u)
            for c in nx.connected_components(G_sample):
                for node in c:
                    connected_components[node] = c
            self.graph_samples.append((G_sample, neighbors, connected_components))

    def submodular_func_caching(self, solution_elements, item_id):
        """
        Submodular function with caching
        :param solution_elements: current solution nodes
        :param item_id: nodes to add
        :return: val, updated_solution_elements
        """
        if not solution_elements and not item_id:
            return 0, []

        spread = []
        counter = 0

        for G, neighbors, connected_components in self.graph_samples:
            key = tuple(solution_elements)
            if key in self.reachable_nodes_memory:
                if counter in self.reachable_nodes_memory[key]:
                    E_S = self.reachable_nodes_memory[key][counter]
                    consider_nodes = item_id
                else:
                    E_S = set()
                    consider_nodes = solution_elements + item_id
            else:
                E_S = set()
                consider_nodes = solution_elements + item_id

            reachable_nodes = []
            for node in consider_nodes:
                if node not in E_S:
                    if node not in connected_components:
                        continue
                    reachable_nodes += connected_components[node]

            reachable_nodes = list(E_S) + reachable_nodes
            E_S = set(reachable_nodes)
            spread.append(len(E_S))

            new_key = tuple(solution_elements + item_id)
            if new_key in self.reachable_nodes_memory:
                if counter not in self.reachable_nodes_memory[new_key]:
                    self.reachable_nodes_memory[new_key][counter] = E_S
            else:
                self.reachable_nodes_memory[new_key] = {}
                self.reachable_nodes_memory[new_key][counter] = E_S

            counter += 1

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
        prev_val, _ = self.submodular_func_caching(current_nodes, [])
        new_val, _ = self.submodular_func_caching(current_nodes, [new_node])
        return new_val - prev_val

    def getNodeInfluenceAdd(self, influence_x, node_index, curr_solution, curr_influence):
        '''
        Helper function to get influence addition of new node
        '''
        marginal_gain = self.compute_marginal_gain(curr_solution, self.nodes[node_index])
        influence_add = curr_influence + marginal_gain
        influence_ratio_add = (min(influence_x, influence_add) - curr_influence) / self.node_costs[self.nodes[node_index]]
        return influence_ratio_add


    def submodularWithBudget(self, influence_x, epsilon_val):
        '''
        Greedy submodular maximization algorithm with knapsack budget from Demaine and Zadimoghaddam 2010
            If there exists an optimal solution with cost at most B and utility at least x, there is polytime
            algorithm that can find a collection of subsets of cost at most O(B log (1/eps)),
            and utility at least (1 - eps) x for any 0 < epsilon < 1
        ARGS:
            influence_x   : minimum desired influence bound
        RETURN:
            solution_node_list    : List of chosen nodes
        '''
        startTime = time.perf_counter()
        
        solution_nodes = []
        curr_influence, curr_cost = 0, 0
        influence_list, cost_list = [0], [0]

        while curr_influence < ((1 - epsilon_val)*influence_x):
            node_max_ratio = 0
            best_node = None

            #Check all nodes, only consider those not in solution
            for i, node in enumerate(self.nodes):
                if node not in solution_nodes:
                    node_ratio = self.getNodeInfluenceAdd(influence_x, i, solution_nodes, curr_influence)

                    if node_ratio > node_max_ratio:
                        best_node = node
                        best_node_cost = self.node_costs[node]
                        node_max_ratio = node_ratio

            #Add best node to solution
            solution_nodes.append(best_node)
            curr_influence = self.compute_influence(solution_nodes)
            curr_cost += best_node_cost
            logging.info("Added node: {} to solution, curr_influence: {:.3f}, curr_cost: {}".format(best_node, curr_influence, curr_cost))

            #Update incremental influence and cost
            influence_list.append(curr_influence)
            cost_list.append(curr_cost)

        logging.info("Final solution: {}, influence: {}, cost: {}".format(solution_nodes, curr_influence, curr_cost))
        self.plotParetoCurve(influence_list, cost_list)

        runTime = time.perf_counter() - startTime
        return solution_nodes, runTime
    

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
        influence_list, cost_list = [0], [0]

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

                #Perform Greedy+ check - Loop over solution in each iteration of plain greedy
                for j, node_j in enumerate(solution_nodes):
                    feasible_nodes = solution_nodes[:j+1]
                    feasible_cost = sum(self.node_costs[node] for node in feasible_nodes)
                    logging.debug("Trying incremental solution:{}, cost:{}".format(feasible_nodes, feasible_cost))

                    for k, node_k in enumerate(self.nodes):
                        #If adding a single node doesn't violate budget
                        if node_k not in feasible_nodes and feasible_cost + self.node_costs[node_k] <= self.B:
                            #Compute influence by adding node to incremental solution
                            added_influence = self.compute_influence(feasible_nodes + [node_k])

                            #If this solution is better than original solution, store it
                            if added_influence > seed_i_influence:
                                seed_i_nodes = feasible_nodes + [node_k]
                                seed_i_influence = added_influence
                                seed_i_cost = feasible_cost + self.node_costs[node_k]
                                logging.debug("New feasible seed solution yielded better influence! {}, influence={:.3f}, cost={}".format(seed_i_nodes,
                                                                                                                                       seed_i_influence, seed_i_cost))

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

    # Take only the largest connected component with size <= 5000
    if len(G_undir) > 0:
        components = [cc for cc in nx.connected_components(G_undir) if len(cc) <= 20000]
        if components:
            largest_cc = max(components, key=len)
        else:
            # If no component <=1200, take the largest overall
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
        # Default uniform costs
        node_costs = {node: 1 for node in G.nodes()}
    
    logging.info("Imported influence graph with {} nodes and {} edges".format(G.number_of_nodes(), G.number_of_edges()))
    
    return G, node_costs