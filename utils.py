import logging, pickle
import numpy as np
logging.basicConfig(format='%(asctime)s |%(levelname)s: %(message)s', level=logging.INFO)

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

    # with open(data_path + 'graphMat_{}.pkl'.format(dataset_num), "rb") as fp:
    #     graphmat = pickle.load(fp)
    #     logging.info("Imported {} graph matrix, Shape: {}\n".format(dataset_name, graphmat.shape))

    return experts, tasks, costs_arr