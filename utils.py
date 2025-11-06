import logging, pickle, json
import numpy as np
logging.basicConfig(format='%(asctime)s |%(levelname)s: %(message)s', level=logging.INFO)


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


def import_pickled_datasets(dataset_name, dataset_num):
    '''
    Code to quickly import final datasets for experiments
    '''
    data_path = 'datasets/pickled_data/' + dataset_name + '/' + dataset_name + '_'
    
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