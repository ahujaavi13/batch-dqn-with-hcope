import torch
import torch.nn.functional as F
import numpy as np

from agent.dqn import DQN


class Agent:
    def __init__(self, num_states, fc1_dim, fc2_dim, num_actions,
                 learning_rate, batch_size, is_training, seed, filename=None, verbose=True):
        """Vanilla DQN modified for Batch Data"""

        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.log_counter = 0
        self.verbose = verbose
        # Agent Hyperparameters

        self.num_states = num_states
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.target_network = DQN(num_states, fc1_dim, fc2_dim, num_actions, seed).to(self.device)
        self.current_network = DQN(num_states, fc1_dim, fc2_dim, num_actions, seed).to(self.device)

        if not is_training:
            self.current_network.load_state_dict(torch.load(filename))

        self.optimizer = torch.optim.Adam(self.current_network.parameters(), lr=self.learning_rate)

        # self.time_steps = 0
        self.learned_policy = np.zeros((self.num_states, num_actions))

    def step(self, experiences, gamma, tau, is_training):
        """Agent can't act in Batch setting. Throws Exception"""
        raise Exception(f"[Agent.step] Agent can't act in Batch setting.")

    def act(self, experiences, is_training):
        """Choose an action based on policy from a state s"""
        state = torch.from_numpy(np.vstack(self.one_hot(experiences, 0))).float()
        s = experiences[0][:, 0].astype(int)
        self.current_network.eval()
        with torch.no_grad():
            action_values = self.current_network(state, is_training)

        if self.verbose and self.log_counter % 5000 == 0:
            print(f"[{self.log_counter}] Test Action Value: {torch.max(action_values)}")

        self.log_counter += 1

        for i, _s in enumerate(s):
            self.learned_policy[_s] += action_values[i].cpu().data.numpy()

    def learn(self, experiences, gamma, tau, is_training):
        """Update policy"""

        self.current_network.train()

        s = torch.from_numpy(np.vstack(self.one_hot(experiences, 0))).float().to(self.device)
        a = torch.from_numpy(np.vstack(self.one_hot(experiences, 1))).long().to(self.device)
        r = torch.from_numpy(np.vstack(experiences[0][:, 2])).float().to(self.device)
        s_ = torch.from_numpy(np.vstack(self.one_hot(experiences, 3))).float().to(self.device)
        _done = torch.from_numpy(np.vstack(experiences[0][:, 4]).astype(np.uint8)).float().to(self.device)

        target_value = torch.from_numpy(self.one_hot_target(self.target_network(s_, is_training).detach()))

        target_value = r + gamma * target_value * (1 - _done)
        current_value = self.current_network(s, is_training).gather(1, a)

        loss = F.mse_loss(current_value, target_value)

        if self.verbose and self.log_counter % 5000 == 0:
            print(f"[{self.log_counter}] Target Action Value: {torch.max(target_value)}, "
                  f"Current Action Value: {torch.max(current_value)}, "
                  f"MSE: {loss}")
        self.log_counter += 1
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._update_weights(tau)

    def _update_weights(self, tau):
        for target_weights, current_weights in zip(self.target_network.parameters(), self.current_network.parameters()):
            target_weights.data.copy_(tau * target_weights.data + (1.0 - target_weights) * current_weights.data)

    def one_hot(self, experiences, index):
        """Convert experiences to one-hot

        :param experiences:
        :param index:
        :return:
        """
        a = experiences[0][:, index].astype(int)
        if index != 1:
            b = np.zeros((a.size, self.num_states))
        else:
            b = np.zeros((a.size, self.num_actions))

        b[np.arange(a.size), a] = 1

        return b

    # noinspection PyMethodMayBeStatic
    def one_hot_target(self, target_value):
        """Covert target to one-hot vector

        :param target_value:
        :return:
        """
        b = np.zeros_like(target_value)
        b[np.arange(len(target_value)), target_value.argmax(1)] = 1

        return b
