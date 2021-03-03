import os
import logging
import numpy as np
import torch

from utils.data_loader import DataLoader
from agent.agent import Agent


class Trainer:
    def __init__(self, conf, load_data, is_logging, start_index=0, end_index=100000):

        self.config = conf

        # Logging Boilerplate
        logging.basicConfig(level=logging.NOTSET)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.is_logging = is_logging

        # Training Hyperparameters
        self.num_episodes = int(1e4)
        self.num_test_eps = int(1e4)
        self.batch_size = 1
        self.learning_rate = 6e-4
        self.gamma = 0.95
        self.tau = 5e-3
        self.fc1_dims = 128
        self.fc2_dims = 128

        # Seed
        self.seed = 42
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Environment variables
        self.num_states = 18
        self.num_actions = 4

        # Load data
        self.dataLoader = DataLoader(self.seed, os.getcwd(), self.config["raw_data"], logging.DEBUG)
        if load_data:
            self.data = self.dataLoader.load_raw_data()
        else:
            self.data = \
                self.dataLoader.load_processed_data(path=os.getcwd(), filename=self.config["dqn_data"])

        self.data = self.data[1:].reshape(-1, 1)[:, 0][start_index:end_index]
        self.logger.info(f" [Trainer] Data Shape: {self.data.shape}")

    def train(self, filename, is_training=True):

        # Initialize Agent
        agent = Agent(
            num_states=self.num_states,
            fc1_dim=self.fc1_dims,
            fc2_dim=self.fc2_dims,
            num_actions=self.num_actions,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            is_training=is_training,
            seed=self.seed,
            filename=filename
        )

        for ep in range(self.num_episodes):
            experiences = self.dataLoader.sample(batch_size=self.batch_size)
            agent.learn(experiences, self.gamma, self.tau, is_training)
            if self.is_logging and (ep+1) % 100 == 0:
                self.logger.debug(f"Trained on episode {ep}")

        torch.save(agent.current_network.state_dict(), filename)
        if self.is_logging:
            self.logger.debug(f" [Trainer.train] Trainer file saved!")

    def test(self, filename, p_name, is_training=False):
        """Test the policy on unseen data to generate probabilities"""
        agent = Agent(
            num_states=self.num_states,
            fc1_dim=self.fc1_dims,
            fc2_dim=self.fc2_dims,
            num_actions=self.num_actions,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            is_training=is_training,
            seed=self.seed,
            filename=filename
        )

        for ep in range(self.num_test_eps):
            experiences = self.dataLoader.sample(batch_size=self.batch_size)
            agent.act(experiences, is_training)
            if self.is_logging and (ep+1) % 100 == 0:
                print(f"Tested on episode {ep}")

        learned_policy = np.array(agent.learned_policy / self.num_test_eps)
        np.savetxt(p_name, learned_policy, delimiter='\n')
        if self.is_logging:
            self.logger.info('[Trainer.test] Policy file saved')


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1)[:, np.newaxis])
    return e_x / e_x.sum(axis=1)[:, np.newaxis]

