import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym

class ExperienceReplayBuffer:
    def __init__(self, max_size: int, environment_name: str, parallel_game_unrolls: int):
        self.max_size = max_size
        self.environment_name = self.environment_name
        self.parallel_game_unrolls = parallel_game_unrolls
        self.envs = gym.vector.make(environment_name, parallel_game_unrolls)

    def fill_with_samples(self, dqn_network):
        observation, _ = self.envs.reset()
        for i in range(10):
            action = self.sample_epsilon_greedy(dqn_network, observation)
            next_observations, reward, terminateds, _, _ = self.env.step(action)
            observations = next_observations

        pass

    def create_dataset(self):
        pass

    def sample_epsilon_greedy(self, dqn_network, observation):
        pass

def dqn():
    ENV_NAME = 'ALE/Breakout-v5'
    ERP_SIZE = 10000
    erp = ExperienceReplayBuffer(ERP_SIZE, ENV_NAME)
    dqn_agent = #TODO
    done = False
    while not done:
        pass