import numpy as np
import time
import matplotlib.pyplot as plt

class GridWorld:

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.agent = [0, 0]
        self.goal = [self.height - 1, self.width - 1]
        self.wall = [3, 2]
        self.trap = [2, 2]
        self.num_actions = 4
        self.actions = [0,1,2,3]
       

    def step(self, direction):

        row, col = self.agent

        #move right
        if direction == 0:
            if col < self.width - 1 and [row, col + 1] != self.wall:
                col += 1

        #move left
        elif direction == 1:
            if col > 0 and [row, col - 1] != self.wall:
                col -= 1
        #move up
        elif direction == 2:
            if row > 0 and [row - 1, col] != self.wall:
                row -= 1

        #move down
        elif direction == 3:
            if row < self.height - 1 and [row + 1, col] != self.wall:
                row += 1

        self.agent = [row, col]

        #check if agent reached the goal
        done = (self.agent == self.goal)

        #give reward
        if self.agent == self.trap:
            reward = -1

        elif self.agent == self.goal:
            reward = 1 
        else: 
            reward = -0.1

        return self.agent, reward, done

    def reset(self):
        self.agent = [0, 0]
        return self.agent

    def evaluate(self):
        #check current position to goal
        row, col = self.agent
        row_goal, col_goal = self.goal
        position_to_goal = [row_goal - row, col_goal - col]

        #check which actions lead closer to the goal
        useful_actions = []
        if position_to_goal[0] > 0: #down
            useful_actions.append(3)

        elif position_to_goal[0] < 0: #up
            useful_actions.append(2)

        if position_to_goal[1] > 0: #right
            useful_actions.append(0)

        elif position_to_goal[1] < 0: #left
            useful_actions.append(1)
        
        return useful_actions
    

    def render(self):
        #just for visualization 
        grid = np.zeros((self.height, self.width), dtype=str)
        grid[self.agent[0], self.agent[1]] = "A"
        grid[self.goal[0], self.goal[1]] = "G"
        grid[self.wall[0], self.wall[1]] = "W"
        grid[self.trap[0], self.trap[1]] = "T"
        print(grid)

    def mc_estimation(self, n_epochs=1000, gamma=0.99, epsilon=0.1, alpha=0.1):    
        # Initialize empty dictionaries to store returns and counts for each state
       
        Q = np.zeros((self.height, self.width,self.num_actions))
        count = np.zeros((self.height, self.width, self.num_actions))
        #get a variable to accumulate the rewards of all epochs
        total_reward = 0 
        average_reward = []
        

        # Run n_epochs epochs
        for epoch in range(n_epochs):
            episode = []
            state = self.reset()
            done = False

            while not done:
                if np.random.rand() < epsilon: # choose the action depending on the soft policy
                    action = np.random.choice(self.actions)

                else: # choose the action with the maximum q-value
                    q_values = Q[state[0], state[1], :]
                    max_q = np.max(q_values)
                    max_indices = np.where(q_values == max_q)[0]
                    action = np.random.choice(max_indices) #if there is two actions with same maximum value then it will choose randomly from these two


                #step and save state, action and reward for mc estimate
                next_state, reward, done = self.step(action)
                episode.append((state, action, reward))
                state = next_state

            
            G = 0 # total discounted return the agent receive after making an action in specific state
            #visited = np.zeros((self.width, self.height), dtype=bool)
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = gamma * G + reward # update the G
                count[state[0], state[1], action] += 1 # update the count
                # update the Q value, N represent the number of times the sate and the action has been visted
                Q[state[0], state[1], action] += alpha * (G - Q[state[0], state[1], action]) / count[state[0], state[1], action]
            total_reward += G
            average_reward.append(total_reward / (epoch + 1))

        # empty list to save the q-values
        policy = np.zeros((self.height, self.width), dtype=int)
        for i in range(self.height):
            for j in range(self.width):
                policy[i, j] = np.argmax(Q[i, j, :]) # adding the q-value to the policy list

        return Q, policy, average_reward
    
    def plot_average_return(self, returns, time, n_epochs=1000):
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
        fig.subplots_adjust(wspace=0.3)
        fig.suptitle('Average Return per Episode')
        ax1.plot(range(n_epochs), returns)
        ax1.set_xlabel('number of episodes')
        ax2.set_xlabel('runtime in seconds')
        ax2.plot(np.linspace(0, time, 1000), returns)
        plt.show()


    def visualize_state_values(self, policy):
        fig, ax = plt.subplots()
        ax.set_xticks(np.arange(self.width))
        ax.set_yticks(np.arange(self.height))
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
        ax.grid(True)
        im = ax.imshow(np.array(policy), cmap='viridis')
        cbar = ax.figure.colorbar(im, ax=ax)
        plt.show()

if __name__ == '__main__':
    env = GridWorld(4, 4)
    start_time = time.time()
    _, policy, average_return = env.mc_estimation()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(policy)
    env.render()
    env.plot_average_return(average_return, elapsed_time)
    
