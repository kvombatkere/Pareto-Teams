import time, json, pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import logging
logging.basicConfig(format='%(asctime)s |%(levelname)s: %(message)s', level=logging.INFO)


def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return np.round(intersection / union, 3)


def getJaccardSimilarities(setList):
    '''
    Given a list of sets, compute Jaccard Similarity of all pairs
    '''
    jaccardSimList= []
    allPairs = [(set(a), set(b)) for idx, a in enumerate(setList) for b in setList[idx+1:]]

    for (a,b) in allPairs:
        jaccardSimList.append(jaccard_similarity(a,b))

    plt.hist(jaccardSimList, bins=20)

    return jaccardSimList


#Compute cardinality of cover
def setCoverageCardinality(setList):
    '''
    Given a list of sets setList, compute the cardinality of the cover
    '''
    union_sets = set()
    for s in setList:
        union_sets = union_sets.union(s)

    return len(union_sets)


#Compute union of list of sets
def setUnion(setList):
    '''
    Given a list of sets setList, compute the union
    '''
    union_sets = set()
    for s in setList:
        union_sets = union_sets.union(s)

    return union_sets


#Greedy Algorithm for Max-K Coverage
def greedyCoverage(setListInput, k):
    '''
    Given a list of sets setList, and an integer k, Greedily find the top-k sets
    '''
    startTime = time.perf_counter()
    setList = setListInput.copy()
   
    #Start with a set with max cardinality - get index of set with max length
    setLengths = [len(s) for s in setList]
    maxLengthIndex = setLengths.index(max(setLengths))
    
    currentSol = [setList.pop(maxLengthIndex)] 

    #Add k-1 more sets Greedily
    for i in range(k-1):
        maxCardinality = 0
        bestGreedySetIndex = 1000
        #Iterate over remaining sets in setList
        for j, candidate_set in enumerate(setList):
            candidateSetList = currentSol + [candidate_set]
            candidateCardinality = setCoverageCardinality(candidateSetList)

            if candidateCardinality > maxCardinality:
                maxCardinality = candidateCardinality
                bestGreedySetIndex = j

        currentSol.append(setList.pop(bestGreedySetIndex))
        #print("i = {} Current Solution: {}".format(i, currentSol))

    greedyObjective = setCoverageCardinality(currentSol)
    logging.info("Greedy Cardinality = {}, Solution: {}".format(greedyObjective, currentSol))

    logging.debug("Computed Greedy Solution, Cardinality: {}".format(greedyObjective))

    runTime = time.perf_counter() - startTime
    logging.debug("Greedy Max-K Cover computation time = {:.3f} seconds".format(runTime))

    return greedyObjective, currentSol


def csgMarginalGain(currentSol_Q, currentCost_Q, newExpert, task, newExpertCost, lambdaVal, currentObjective):
    '''
    Compute the scaled objective
    '''
    newSol = currentSol_Q.union(set(newExpert))
    newSolCoverage = lambdaVal * len(newSol.intersection(set(task)))

    #Compute new scaled objective
    currentCost_Q += 2*newExpertCost
    scaledObjective = newSolCoverage - currentCost_Q

    marginalGain = scaledObjective - currentObjective
    return marginalGain


#Cost Scaled Greedy Algorithm for Cardinality Constrained
def costScaledGreedyCoverage(experts, task, costs, k, lambdaVal):
    '''
    Given a list of experts, a task, and costs, use Cost-Scaled Greedy to find solution
    '''
    startTime = time.perf_counter()

    logging.info("Computing CSG Greedy for Task: {}".format(task))
    expertList = experts.copy()
    costsArr = costs.copy()
    currentSolExp = [] #keep track of experts in solution
    currentSolElements = set() #track elements in solution
    
    #Initialize cost and objective
    prev_objective = 0
    currentCost = 0    

    #Add sets Greedily upto cardinality k
    for i in range(k):
        #print("Current Experts: {}, Elements: {}".format(currentSolExp, currentSolElements))

        bestMarginalGain = 0
        bestGreedySetIndex = -1000
        #Iterate over remaining sets in setList
        for j, candidate_expert in enumerate(expertList):
            expert_marginalGain = csgMarginalGain(currentSolElements, currentCost, candidate_expert, 
                                                  task, costsArr[j], lambdaVal, prev_objective)

            if expert_marginalGain > bestMarginalGain:
                bestMarginalGain = expert_marginalGain     
                bestGreedySetIndex = j

        if bestMarginalGain == 0:
            #print("No more marginal gain from adding experts: break")
            break
        
        #Update solution, objective and costs
        #print("Num Experts remaining: {}".format(len(expertList)))
        expertChosen = expertList.pop(bestGreedySetIndex)
        expertCost = costsArr.pop(bestGreedySetIndex)
        
        currentSolElements = currentSolElements.union(set(expertChosen))
        currentSolExp.append(expertChosen)
        currentCost += 2*expertCost

        prev_objective += bestMarginalGain #add marginal gain to objective

    #compute coverage
    solutionCoverage = ((prev_objective + currentCost)/lambdaVal)/len(task)

    unscaledObjective = prev_objective + (currentCost/2)
    logging.info("Computed CSG Solution: {}, Objective: {}, Number of Experts: {}, Cost: {}, Coverage: {:.3f}".format(currentSolExp, unscaledObjective, 
                                                                                                           len(currentSolExp), currentCost/2, solutionCoverage))

    runTime = time.perf_counter() - startTime
    logging.debug("Greedy computation time = {:.4f} seconds".format(runTime))

    return unscaledObjective, solutionCoverage, len(currentSolExp), currentCost/2


##Randomized solution
def chooseRandomExperts(experts, task, costs, k, lambdaVal):
    '''
    Given a set of experts, corresponding costs, and an integer k pick k experts randomly
    '''
    startTime = time.perf_counter()

    logging.debug("Computing Random Solution for Task: {}".format(task))

    randomExpertIndices = np.random.choice(len(experts), size=k, replace=False)
    randomSolExperts = [experts[indx] for indx in randomExpertIndices]
    randomSolElements = set() #track elements in solution
    randomSolCost = 0

    for exp in randomSolExperts:
        indx_exp = experts.index(exp)
        randomSolCost += costs[indx_exp]
        randomSolElements = randomSolElements.union(set(exp))

    #compute coverage
    randomSolutionCoverage = len(randomSolElements.intersection(set(task)))/len(task)
    randomObjective = lambdaVal*len(randomSolElements.intersection(set(task))) - randomSolCost
    logging.info("Baseline Random Solution: {}, Objective: {:.2f}, Number of Experts: {}, Cost: {}, Coverage: {:.3f}".format(randomSolExperts, randomObjective, 
                                                                                                           len(randomSolExperts), randomSolCost, randomSolutionCoverage))

    runTime = time.perf_counter() - startTime
    logging.debug("Random computation time = {:.4f} seconds".format(runTime))

    return randomObjective, randomSolutionCoverage, randomSolCost


#Randomized solution for Coverage - Edge Weights
def randomExpertsEdgeWeights(experts, task, edge_weights_matrix, k, lambdaVal):
    '''
    Given a set of experts, corresponding edge weight matrix, and an integer k pick k experts randomly
    '''
    startTime = time.perf_counter()

    logging.debug("Computing Random Cov-Edges Solution for Task: {}".format(task))

    randomExpertIndices = np.random.choice(len(experts), size=k, replace=False)
    randomSolExperts = [experts[indx] for indx in randomExpertIndices]
    randomSolElements = set() #track elements in solution
    edgeWeightCost = 0

    #Compute expert edge weight total cost of each pair in assignment
    for i, expert_1 in enumerate(randomSolExperts):
        exp_1_indx = experts.index(expert_1)
        randomSolElements = randomSolElements.union(set(expert_1)) #Add expert skills to the solution
        for j, expert_2 in enumerate(randomSolExperts[i+1:]):
            exp_2_indx = experts.index(expert_2)
            edgeWeightCost += edge_weights_matrix[exp_1_indx][exp_2_indx]

    #compute coverage
    randomSolutionCoverage = len(randomSolElements.intersection(set(task)))/len(task)
    randomObjective = lambdaVal*len(randomSolElements.intersection(set(task))) - edgeWeightCost
    logging.info("Baseline Random Solution: {}, Objective: {:.2f}, Number of Experts: {}, Cost: {}, Coverage: {:.3f}".format(randomSolExperts, randomObjective, 
                                                                                        len(randomSolExperts), edgeWeightCost, randomSolutionCoverage))

    runTime = time.perf_counter() - startTime
    logging.debug("Random computation time = {:.4f} seconds".format(runTime))

    return randomObjective, randomSolutionCoverage, edgeWeightCost


def baselineTopKCosts(experts, task, costs, k, lambdaVal):
    '''
    Baseline Algorithm to choose top k experts with maximum overlap
    '''
    #Get intersection size of each expert
    intersectionDict = {}
    for i, expert_i in enumerate(experts):
        intersectionDict[i] = set(expert_i).intersection(set(task))

    sortedExpertIndices = sorted(intersectionDict, key=intersectionDict.get, reverse=True)

    topKExperts = []
    topKSolElements = set()
    topKCost = 0

    for j in range(k):
        expertIndx_j = sortedExpertIndices[j]
        topKExperts.append(experts[expertIndx_j])
        topKSolElements = topKSolElements.union(set(experts[expertIndx_j]))
        topKCost += costs[expertIndx_j]

    solutionCoverage = len(topKSolElements.intersection(set(task)))/len(task)
    solutionObjective = lambdaVal*len(topKSolElements.intersection(set(task))) - topKCost
    logging.info("Baseline Top-K Solution:{}, Objective: {:.2f}, Number of Experts: {}, Cost: {}, Coverage: {:.2f}".format(topKExperts, solutionObjective, 
                                                                                        len(topKExperts), topKCost, solutionCoverage))
    
    return solutionObjective, solutionCoverage, topKCost


def baselineTopKEdgeWeights(experts, task, edge_weights_matrix, k, lambdaVal):
    '''
    Baseline Algorithm to choose top k experts with maximum overlap
    '''
    #Get intersection size of each expert
    intersectionDict = {}
    for i, expert_i in enumerate(experts):
        intersectionDict[i] = set(expert_i).intersection(set(task))

    sortedExpertIndices = sorted(intersectionDict, key=intersectionDict.get, reverse=True)
    topKExperts = []
    topKSolElements = set()

    for j in range(k):
        expertIndx_j = sortedExpertIndices[j]
        topKExperts.append(experts[expertIndx_j])
        topKSolElements = topKSolElements.union(set(experts[expertIndx_j]))

    #Compute expert edge weight total cost of each pair in assignment
    edgeWeightCost = 0
    for i, expert_1 in enumerate(topKExperts):
        exp_1_indx = experts.index(expert_1)
        for j, expert_2 in enumerate(topKExperts[i+1:]):
            exp_2_indx = experts.index(expert_2)
            edgeWeightCost += edge_weights_matrix[exp_1_indx][exp_2_indx]

    solutionCoverage = len(topKSolElements.intersection(set(task)))/len(task)
    solutionObjective = lambdaVal*len(topKSolElements.intersection(set(task))) - edgeWeightCost
    logging.info("Baseline Top-K Solution:{}, Objective: {:.2f}, Number of Experts: {}, EdgeCost: {}, Coverage: {:.2f}".format(topKExperts, solutionObjective, 
                                                                                        len(topKExperts), edgeWeightCost, solutionCoverage))
    
    return solutionObjective, solutionCoverage, edgeWeightCost


#Import Datasets
def importData(experts_filename, tasks_filename, numExperts=1000, exp_len=2, task_len=4):
    with open(experts_filename, 'r') as f:
        experts_list = json.loads(f.read())
        expert_list_int = []
        expert_index_list = []
        for i, expert_skills in enumerate(experts_list[:numExperts]):
            exp_skillset = sorted([int(skill) for skill in expert_skills])
            #only keep unique experts with at least exp_len skills
            if len(exp_skillset) >= exp_len and exp_skillset not in expert_list_int and len(expert_list_int) <= numExperts:
                expert_list_int.append(exp_skillset)
                expert_index_list.append(i)

    with open(tasks_filename, 'r') as f:
        tasks_list = json.loads(f.read())
        task_list_int = []
        for task_skills in tasks_list:
            task_skills = sorted([int(skill) for skill in task_skills])
            #Keep only unique tasks with at least task_len skills
            if len(task_skills) >= task_len and task_skills not in task_list_int:
                task_list_int.append(task_skills)

    logging.info("Imported {} and {}. Num Experts={}, Num Tasks={}".format(experts_filename.split('/')[-1], tasks_filename.split('/')[-1],
                                                                    len(expert_list_int),len(task_list_int)))
    
    return task_list_int, expert_list_int, expert_index_list


#Reduce number of skills in a dataset
def reduceSkills(input_tasks, input_experts, num_skills=50):
    '''
    Given an input set of tasks and experts, reduce number of skills down to num_skills
    Note that skills must be indexed 0,1,2,..
    '''
    reduced_skills_tasks, reduced_skills_experts = [], []

    for task in input_tasks:
        task_reduced = []
        for skill in task:
            task_reduced.append(skill % num_skills)
        reduced_skills_tasks.append(list(set(task_reduced)))

    for exp in input_experts:
        expert_reduced = []
        for skill in exp:
            expert_reduced.append(skill % num_skills)
        reduced_skills_experts.append(list(set(expert_reduced)))

    logging.info("Reduced input task and experts to {} skills".format(num_skills))
    return reduced_skills_tasks, reduced_skills_experts


def importGraphData(filename, expert_indices):
    '''
    Import graph data and add edge weights
    Return a symmetric edgeweight matrix
    '''
    with open(filename, "rb") as fp:
        adjMat = pickle.load(fp)
        adjMat = adjMat[expert_indices,:][:,expert_indices]
        logging.info("Imported {} graph matrix: {}\n".format(filename.split('/')[-1], adjMat.shape))

    #Values to populate edge weights
    edgeWeightVals = [100, 101, 102, 103, 104]
    edgeWeightMatrix = np.zeros((len(expert_indices), len(expert_indices)))

    for i in range(len(adjMat)):
        for j in range(len(adjMat[i])):
            if i != j:
                #If not 1, copy value over
                if adjMat[i][j] != 1:
                    edgeWeightMatrix[i][j] = adjMat[i][j]*10

                #If value is 1, assign a new random edge weight
                else:
                    edgeWeightMatrix[i][j] = np.random.choice(edgeWeightVals)

    return edgeWeightMatrix


#Plot pairwise jaccard similarity histogram between task pairs
def plot_pairwise_jaccard_distribution(task_list):
    '''
    Calculate jaccard similarites between all pairs of tasks in task_list
    Plot distribution as histogram
    '''
    task_similarities = []
    for i, task_i in enumerate(task_list):
        for j, task_j in enumerate(task_list[i+1:]):
            taskSim = jaccard_similarity(set(task_i), set(task_j))
            task_similarities.append(taskSim)

    logging.info('Computed {} pairwise Jaccard Similarities between {} tasks'.format(len(task_similarities), len(task_list)))
    
    fig_hist = plt.figure(figsize=(8,3))
    plt.hist(task_similarities, bins=20)
    plt.xlabel("Jaccard Similarity")
    plt.ylabel("Frequency of Task Pairs")
    plt.grid()
    plt.show()

    return None


def import_pickled_datasets(dataset_name, dataset_num):
    '''
    Code to quickly import final datasets for experiments
    '''
    data_path = '../datasets/pickled_data/' + dataset_name + '/' + dataset_name + '_'
    
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


def getDatasetSummary(task_list, expert_list, edgeWeightMat, datasetName):
    distinct_skills = []
    total_skills_tasks = 0
    for task in task_list:
        total_skills_tasks += len(task)
        for sk in task:
            if sk not in distinct_skills:
                distinct_skills.append(sk)

    total_skills_exp = 0
    for exp in expert_list:
        total_skills_exp += len(exp)
        for sk in exp:
            if sk not in distinct_skills:
                distinct_skills.append(sk)

    avg_skills_tasks = total_skills_tasks/len(task_list)
    avg_skills_exp = total_skills_exp/len(expert_list)

    print("Dataset:{}, Skills/Expert = {}, Skills/Task = {}, Distinct skills = {}".format(datasetName, avg_skills_exp, avg_skills_tasks, len(distinct_skills)))

    #Get average unweighted degree
    adjMat_plot = edgeWeightMat.copy()
    adjMat_plot[adjMat_plot >= 100] = 0
    G_imdb = nx.from_numpy_array(adjMat_plot, parallel_edges=False)

    node_degrees = list(G_imdb.degree())

    weights = 0
    for n,d in node_degrees:
        weights += d

    avg_degree = weights/len(node_degrees)
    print("Average Degree of Graph:{:.2f}".format(avg_degree))

    ccs = [len(c) for c in sorted(nx.connected_components(G_imdb), key=len, reverse=True)]
    print("Number of Connected Components: {}\nSize of Largest CC: {}".format(len(ccs), ccs[0]))

    largest_cc = sorted(nx.connected_components(G_imdb), key=len, reverse=True)[0]
    CC_subgraph = G_imdb.subgraph(largest_cc).copy()
    print("Average Shortest Path Length of Largest CC: {:.2f}".format(nx.average_shortest_path_length(CC_subgraph, weight=None)))
    return None