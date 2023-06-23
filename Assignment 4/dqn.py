import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym

class ExperienceReplayBuffer:
    def __init__(self, max_size: int, environment_name: str, parallel_game_unrolls: int, observation_preprocessing_function: callable, unroll_steps: int):
        self.max_size = max_size
        self.environment_name = self.environment_name
        self.parallel_game_unrolls = parallel_game_unrolls
        self.observation_preprocessing_function = observation_preprocessing_function
        self.unroll_steps = unroll_steps
        self.envs = gym.vector.make(environment_name, parallel_game_unrolls)
        self.num_possible_actions = self.single_action_space.n
        self.current_states, _ = self.envs.reset()
        self.data = []

    def fill_with_samples(self, dqn_network, unroll_steps: int):
        states_list = []
        actions_list = []
        rewards_list = []
        seubsequent_states_list = []
        terminateds_list = []

        for i in range(self.unroll_steps):
            actions = self.sample_epsilon_greedy(dqn_network)
            next_states, rewards, terminateds, _, _ = self.env.step(actions)
            states_list.append(self.current_states)
            actions_list.append(actions)
            rewards_list.append(rewards)
            seubsequent_states_list.append(next_states)
            terminateds_list.append(terminateds)
            self.current_states = next_states

        def data_generator():
            for states_batch, actions_batch, rewards_batch, subsequent_states_batch, terminateds_batch in zip(states_list, actions_list, rewards_list, seubsequent_states_list, terminateds_list):
                for game_idx in range(self.parallel_game_unrolls):
                    state = states_batch[game_idx, :, :, :]
                    action = actions_batch[game_idx]
                    reward = rewards_batch[game_idx]
                    subsequent_state = subsequent_states_batch[game_idx, :, :, :]
                    terminated = terminateds_batch[game_idx]

        
        dataset_tensor_specs = (tf.TensorSpec(shape=(210, 160, 3), dtype=tf.uint8), tf.TensorSpec(shape=(), dtype=tf.uint32), tf.TensorSpec(shape=(), dtype=tf.float32), tf.TensorSpec(shape=(210, 160, 3), dtype=tf.uint8), tf.TensorSpec(shape=(), dtype=tf.bool))
        new_samples_dataset = tf.data.Dataset.from_generator(data_generator(), dataset_tensor_specs)

        new_samples_dataset = new_samples_dataset.map(lambda state, action, reward, subsequent_state, terminated: (self.observation_preprocessing_function(state), action, reward, self.observation_preprocessing_function(subsequent_state), terminated)).cache()
        new_samples_dataset = new_samples_dataset.prefetch.cache().shuffle(buffer_size=self.unroll_steps * self.parallel_game_unrolls, reshuffle_each_iteration=True)
        #make sure cache is applied
        for elem in new_samples_dataset:
            continue

        self.data.append(new_samples_dataset)
        datapoints_in_data = len(self.data) * self.unroll_steps * self.parallel_game_unrolls
        if datapoints_in_data > self.max_size:
            self.data.pop(0)
        #return new_samples_dataset

    def create_dataset(self):

        erp_dataset = tf.data.Dataset.sample_from_datasets(self.data, weights = [1/len(self.data) for _ in self.data], stop_on_empty_dataset=False)
        return erp_dataset

    def sample_epsilon_greedy(self, dqn_network, epsilon: float):
        observations = self.observation_preprocessing_function(self.current_states)
        q_values = dqn_network(observations) #tensor of type tf.float32, shape(parallel_gam_unrolls, num_actions)
        greedy_actions = tf.argmax(q_values, axis=1) #shape(parallel_game_unrolls)
        random_actions = tf.random.uniform(shape=(self.parallel_game_unrolls, ), minval = 0, maxval = self.num_possible_actions, dtype=tf.int64)
        epsilon_sampling = tf.random.uniform(shape=(self.parallel_game_unrolls,), minval=0, maxval=1, dtype=tf.float32) > epsilon
        actions = tf.where(epsilon_sampling, greedy_actions, random_actions)
        return actions

def observation_preprocessing_function(observation):
    observation = tf.image.resize(observation, size=(84, 84), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    observation = tf.cast(observation, dtype=tf.float32) / 128.0 - 1.0
    return observation

def create_dqn_network(num_actions: int):
    input_layer = tf.keras.Input(shape=(84, 84, 3), dtype=tf.float32)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu')(input_layer)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu')(input_layer) + x #residual connection
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu')(input_layer) + x
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(input_layer)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(input_layer) + x #residual connection
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(input_layer) + x
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')(input_layer)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')(input_layer) + x #residual connection
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')(input_layer) + x
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')(input_layer)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')(input_layer) + x #residual connection
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')(input_layer) + x
    x = tf.keras.layers.GlobalAvgPool2D()(x)

    x = tf.keras.layers.Dense(units=64, activation='relu')(x) + x
    x = tf.keras.layers.Dense(units=num_actions, activation='linear')(x)
    model = tf.keras.Model(inputs=input_layer, outputs= x)
    return model


def dqn():
    ENV_NAME = 'ALE/Breakout-v5'
    ERP_SIZE = 100000
    PARALLEL_GAME_UNROLLS = 128
    UNROLL_STEPS = 4
    EPSILON = 0.2
    erp = ExperienceReplayBuffer(max_size=ERP_SIZE, environment_name=ENV_NAME, parallel_game_unrolls=PARALLEL_GAME_UNROLLS, observation_preprocessing_function=observation_preprocessing_function, unroll_steps=UNROLL_STEPS)
    dqn_agent = create_dqn_network(num_actions=4)
    done = False
    while not done:
        erp.fill_with_samples(dqn_agent, EPSILON)
        dataset = erp.create_dataset()