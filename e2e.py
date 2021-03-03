import os
import logging

import numpy as np

from utils.data_loader import DataLoader
from train import Trainer
import utils.hcope as H


def loadData(conf, start_index=0, end_index=100000, load_data=False):
    """Load the data from start_index to end_index"""
    dataLoader = DataLoader(0, os.getcwd(), config["raw_data"], logging.DEBUG)
    if load_data:
        X = dataLoader.load_raw_data()
    else:
        X = dataLoader.load_processed_data(path=os.getcwd(), filename=conf["processed_data"])

    print(end_index)
    X = X[1:].reshape(-1, 1)[:, 0][start_index:end_index]
    print(X.shape)
    return X


if __name__ == "__main__":

    config = {
        "raw_data": "sample_data.csv",
        "dqn_data": "data_dqn.npy",
        "processed_data": "data_processed.npy"
    }
    np.random.seed(42)

    os.mkdir('./policy')
    os.mkdir('./policies')
    os.mkdir('./params')

    num_policies = 100
    s_index, e_index = 0, 500000
    trainer = Trainer(config, load_data=False, is_logging=False, start_index=s_index, end_index=e_index)

    # Train and Test DQN, output policies and params
    for it in range(num_policies):
        weights_filename = f'./params/dqn{it + 1}.pth'
        policy_name = f'./policy/policy{it + 1}.txt'
        trainer.train(filename=weights_filename)
        trainer.test(filename=weights_filename, p_name=policy_name)
        trainer.logger.info(f"Training and testing done on file {it + 1}")

    s_index, e_index = 500000, 550000
    X = loadData(config, s_index, e_index, load_data=False)
    np.set_printoptions(precision=5, suppress=True)

    # HCOPE
    c = 2
    delta = 0.01

    # Test for confidence on the remaining data
    for it in range(num_policies):
        candidateSolution = np.loadtxt(f"./policy/policy{it + 1}.txt", delimiter='\n').flatten()

        lowerBound, found = H.safetyTest(candidateSolution, X, delta, c)
        print(f"Iterations: {it}, Start_index: {s_index}, End_index: {e_index}")
        if found:
            print(f"Solution found, lower bound is {lowerBound}")
            np.savetxt(f"./policies/policy{it + 1}.txt", candidateSolution, delimiter='\n')
        else:
            print(f"No solution found, lower bound is {lowerBound}")
            np.savetxt(f"./policies/policy_nsf{it + 1}.txt", candidateSolution, delimiter='\n')

        if (it + 1) % 10 == 0:
            s_index = e_index
            e_index = e_index + 50000
            X = loadData(s_index, e_index)
