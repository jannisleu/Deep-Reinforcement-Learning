import numpy as np
import random
import matplotlib as plt

class GridWorld:

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.agent = [0, 0]
        self.goal = [self.height - 1, self.width - 1]
        self.wall = [1, 1]
        self.trap = [2, 2]

    def step(self, direction):

        x, y = self.agent

        #move right
        if direction == 0:
            if y < self.width - 1 and [x, y + 1] != self.wall:
                y += 1

        #move left
        elif direction == 1:
            if x < self.height - 1 and [x + 1, y] != self.wall:
                y -= 1
        #move up
        elif direction == 2:
            if y > 0 and [x, y - 1] != self.wall:
                x += 1

        #move down
        elif direction == 3:
            if x > 0 and [x - 1, y] != self.wall:
                x -= 1

        self.agent = [x, y]

        #check if agent reached the goal
        done = (self.agent == self.goal)

        #give reward
        if self.agent == self.trap:
            reward = -1.1

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
        x, y = self.agent
        x_goal, y_goal = self.goal
        position_to_goal = [x_goal - x, y_goal - y]

        #check which actions lead closer to the goal
        useful_actions = []
        if position_to_goal[0] > 0:
            useful_actions.append(3)

        elif position_to_goal[0] < 0:
            useful_actions.append(2)

        if position_to_goal[1] > 0:
            useful_actions.append(0)

        elif position_to_goal[1] < 0:
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
    

    def mc_estimation(self, n_epochs=1000):    
        # Initialize empty dictionaries to store returns and counts for each state
        returns = np.zeros((self.height, self.width))
        count = np.zeros((self.height, self.width))

        # Run n_epochs epochs
        for epoch in range(n_epochs):
            # Generate an episode by following the agent's policy until the episode terminates
            episode = []
            state = self.reset()
            done = False

            while not done:
                #apply policy: move closer to goal with prob of 0.8, else move random
                decision = random.choices([0, 1], [0.2, 0.8])
                if decision[add here] == 1: # here you are comparing list to number so this condition will always be false, but i don't know what to add here as i don't know what exactly you are comparing
                    direction = random.choice(self.evaluate())

                else:
                    direction = random.choice([0, 1, 2, 3])

                #step and save state, action and reward for mc estimate
                next_state, reward, done = self.step(direction)
                episode.append((state, direction, reward))
                state = next_state

            # Update returns and counts for each state visited in the episode
            visited = np.zeros((self.width, self.height), dtype=bool)
            for t, (state, direction, reward) in enumerate(episode):
                x, y = state
                if not visited[x, y]:
                    visited[x, y] = True
                    G = sum(r for _, _, r in episode[t:])
                    returns[x, y] += G
                    count[x, y] += 1

        # Calculate the MC estimation for each state
        mc_estimates = returns / count + 1e-6

        return mc_estimates
    
    def visualize_state_values(env, mc_estimates):
        fig, ax = plt.subplots()
        ax.set_xticks(self.width)
        ax.set_yticks(self.height)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)
        im = ax.imshow(np.max(mc_estimates, axis=2), cmap='game')
        cbar = ax.figure.colorbar(im, ax)

if __name__ == '__main__':
    env = GridWorld(4, 4)

    mc_estimates = env.mc_estimation()
    print(mc_estimates)


