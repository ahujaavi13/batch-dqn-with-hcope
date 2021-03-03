import pandas
import os
import logging
import time
import numpy as np


class DataLoader:
    def __init__(self, seed, path=os.getcwd(), filename="sample_data.csv", logging_level=logging.DEBUG):
        """Data loader for DQN and HCOPE"""

        # Logging Boilerplate
        logging.basicConfig(level=logging.NOTSET)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)

        # seed
        self.seed = seed
        np.random.seed(self.seed)

        # Set file path
        self.path = path
        self.filename = filename
        self.SEPARATOR = ','

        # Data Details
        self.total_episodes = int(10e6)
        self.x = np.ndarray([])
        self.y = np.ndarray([])

    def load_raw_data(self):
        """Load All Data"""
        df = pandas.read_csv(self.path + "/" + self.filename, header=[0], delimiter=' ')
        self.logger.info(f"[DataLoader.load] File imported.")
        self.__process_for_dqn__(df)
        self.save_data()
        return self.x

    def load_processed_data(self, path=os.getcwd(), filename="data_dqn.npy"):
        assert os.path.exists(path + "/data/" + filename), f"[DataLaoder.load_processed_data] Path: " \
                                                        f"{path + '/' + filename} does not exist"
        start = time.time()
        self.x = np.load(path + "/data/" + filename, allow_pickle=True)
        self.logger.info(f" Data loaded in {(time.time() - start) : 0.2f} seconds")

        return self.x

    def save_data(self, path=os.getcwd(), filename="data_dqn.npy"):
        np.save(path + "/" + filename, self.x)

    def get(self):
        """Returns data for all the episodes"""
        return self.x

    def sample(self, batch_size=1):
        """Samples data of batch_size where batch_size = number of episodes"""
        experience_indices = np.random.choice(range(self.x.size), batch_size)
        experiences = np.take(self.x, experience_indices)
        self.x = np.delete(self.x, experience_indices)
        return experiences

    def split(self, training_size, test_size):
        """Splits data between training and testing"""
        np.random.shuffle(self.x)
        training_size = self.total_episodes * training_size // 100
        test_size = self.total_episodes * test_size // 100
        self.x, self.y = self.x[:training_size], self.x[test_size:]

    def __process_for_dqn__(self, df):
        """Process data to Num_Episodes x Time_steps x Num_cols in [s, a, r, s_prime, done].
        Used to feed dqn. May need changes depending on data input
        """

        self.logger.info(f" Processing Data...")
        self.logger.debug(f"[DataLoader.__process__] Dataframe shape = {df.shape}")
        start = time.time()
        current_time, total_time = 0, -1
        episodic_data = np.ndarray([])
        ep_count = 0
        data = []
        for index, row in df.iterrows():
            if current_time < total_time:
                episodic_data[current_time] = np.array(row[0].split(self.SEPARATOR) + [0.0], dtype=np.float)
                episodic_data[current_time - 1][-2] = episodic_data[current_time][0]
                current_time += 1
            else:
                if ep_count:
                    episodic_data[current_time - 1][-1] = 1.0
                    episodic_data[current_time - 1][-2] = episodic_data[current_time - 1][0]
                    data.append(episodic_data)
                total_time = int(row[0])
                current_time = 0
                episodic_data = np.ndarray((total_time, 5), dtype=np.float)
                if ep_count and ep_count % 2e3 == 0:
                    self.logger.info(f"Data Processed for {ep_count} episodes")
                ep_count += 1

        self.x = np.asarray(data)
        self.logger.debug(f"[DataLoader.__process__] Data processed in {(time.time() - start) : .2f} secs")

    def __process_for_hcope__(self, df):
        """Process data to Num_Episodes x Time_steps x Num_cols in [s, a, r, pi(a)].
        Used for High Confidence off-policy evaluation. May need changes depending on data input
        """

        self.logger.info(f" Processing Data...")
        self.logger.debug(f"[DataLoader.__process__] Dataframe shape = {df.shape}")
        start = time.time()
        current_time, total_time = 0, -1
        episodic_data = np.ndarray([])
        ep_count = 0
        data = []
        for index, row in df.iterrows():
            if current_time < total_time:
                episodic_data[current_time] = np.array(row[0].split(self.SEPARATOR), dtype=np.float)
                current_time += 1
            else:
                if ep_count:
                    data.append(episodic_data)
                total_time = int(row[0])
                current_time = 0
                episodic_data = np.ndarray((total_time, 4), dtype=float)
                if ep_count and ep_count % 1e3 == 0:
                    self.logger.info(f"Data Processed for {ep_count} episodes")
                ep_count += 1

        self.x = np.asarray(data)
        self.logger.debug(f"[DataLoader.__process__] Data processed in {(time.time() - start) : .2f} secs")
