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
        new_samples_dataset = new_samples_dataset.cache().shuffle(buffer_size=self.unroll_steps * self.parallel_game_unrolls, reshuffle_each_iteration=True)
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
        actions = tf.where(epsilon_sampling, greedy_actions, random_actions).numpy()
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


def train_dqn(train_dqn_network, target_network, dataset, optimizer, num_training_steps: int, gamma: float):
    def training_step(q_target, observations, actions):
        with tf.GradientTape() as Tape:
            q_predictions_all_actions = train_dq_network(observations)
            q_predictions = tf.gather(q_predictions_all_actions, actions, batch_dims = 1)
            loss = tf.reduce_mean(tf.square(q_predictions - q_target))
        geadients = tape.gradient(loss, train_dqn_network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, train_dqn_network.trainable_variables))
        return loss.numpy()
    losses = []
    q_values = []
    for i, state_transition in enumerate(dataset):
        state, action, reward, subsequent_state, terminated = state_transition
        q_vals = target_network(subsequent_state)
        q_values.append(q_vals.numpy())
        max_q_values = tf.reduce_max(q_vals, axis = 1)
        use_subsequent_state = tf.where(terminated, tf.zeros_like(max_q_values, dtype = tf.float32), tf_ones_like(max_q_values, dtype = tf.float32))
        q_target = reward + (gamma * max_q_values * use_subsequent_state)
        loss = training_step(q_target, observations = state)
        losses.append(loss)
        if i >= num_training_steps:
            break
    return np.mean(losses), np.mean(q_values)

def test_q_network(test_dqn_nqtwork, target_network, enviroment_name:str, num_parallel_tests : int, gamma:float):
    envs = gym.vector.make(enviroment_name, parallel_game_unrolls)
    states, _ = envs.rest()
    done = False
    time_step = 0
    episode_finished = np.zeros(num_parallel_tests, dtype = bool)
    returns = np.zeros(num_parallel_tests)
    while not done:
        q_values = test_dqn_network(states)
        actions = tf.argmax(q_values, axis = 1)
        states, rewards, terminateds, _, _ = envs.step(actions)
        episode_finished = np.logical_or(episode_finished, terminateds)
        returns *= ((gamma ** timestep) + rewards) * (np.logical_not(episode_finished).astype(np.float32))
        time_step += 1
        done = np.all(episode_finished)
    return np.mean(returns)

def polyak_averaging_weights(source_network, target_network, polyak_averaging_factor:float):
    source_network_weights = source_network.get_weights()
    target_network_weights = target_network.getweights()
    averaged_weights =[]
    for source_networks_weight, target_network_weight in zip(source_networks_weights, target_network_weights):
        fraction_kept_weights = polyak_averaging_factor * target_network_weight
        fraction_updated_weights = (1- polyak_averaging_factor) * source_network_weight
        averaged_weight = fraction_kept_weights + fraction_updated_weights
        average_weights.append(average_weight)
    target_network.set_weights(averaged_weights)


def dqn():
    ENV_NAME = 'ALE/Breakout-v5'
    NUMBER_ACTIONS = gym.make(ENVIRONMENT_NAME).action_space.n
    ERP_SIZE = 100000
    PARALLEL_GAME_UNROLLS = 128
    UNROLL_STEPS = 4
    EPSILON = 0.2
    GAMMA = 0.98
    PREFILL_STEPS = 100
    POLYAK_AVERAGING_FACTOR = 0.99
    NUM_TRAINING_STEPS_PER_ITER = 4
    NUM_TRAINING_ITERS = 50000
    TEST_EVERY_n_STEPS = 1000
    TEST_NUM_PARELLEL_ENVS = 1024
    erp = ExperienceReplayBuffer(max_size=ERP_SIZE, environment_name=ENV_NAME, parallel_game_unrolls=PARALLEL_GAME_UNROLLS, observation_preprocessing_function=observation_preprocessing_function, unroll_steps=UNROLL_STEPS)
    dqn_agent = create_dqn_network(num_actions=NUMBER_ACTIONS)
    target_network = create_dqn_network(num_actions = NUMBER_ACTIONS)
    dqn_agent.summary()
    dqn_agent(tf.random.uniform(shapes = (1, 84, 84, 3)))
    polyak_averaging_weights(source_network = dqn_agent, target_network = target_network, polyak_averaging_factor = 0.0 )

    dqn_optimizer = tf.keras.optimizer.Adam()
    


    return_tracker = []
    dqn_prediction_error = []
    average_q_values = []
    prefill_exploration_epsilon = 1.8
    for prefill_step in range(PREFILL_STEPS):
        erp.fill_with_samples(dqn_agent, prefill_exploration_epsilon)

    for step in range (NUM_TRAINING_ITERS):
        erp.fill_with_samples(dqn_agent, EPSILON)
        dataset = erp.create_dataset()
        average_loss, average_q_values = train_dqn(dqn_agent, dataset, dqn_optimizer, num_training_steps = NUM_TRAINING_STEPS_PER_ITER, gamma = GAMMA)
        polyak_averaging_weights(source_network = dqn_agent, target_network = target_network, polyak_averaging_factor = POLYAK_AVEERAGING_FACTOR)
        if step % TEST_EVERY_n_STEPS == 0:
            average_return = test_q_network(dqn_agent, ENVIROMENT_NAME, num_parallel_tests, gamma= GAMMA)
            return_tracker.append(average_return)
            dqn_prediction_error.append(average_loss)
            average_q_values.append(average_q_values)
            print(f'TESTING: Average return:{average_return}, Average loss: {average_loss}, Average q-values-estimation:{average_q_values}')
    results_dict = {'average_return': return_tracker, 'average_loss': dqn_prediction_error, 'average_q_values': average_q_values}
    results_df = pd.DataFrame(results_dict)
    visualize_results(results_df, step)
    print(results_df)

  
        

if __name__ == '__main__':
    dqn()
