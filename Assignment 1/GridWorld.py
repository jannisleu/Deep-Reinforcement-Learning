import numpy as np
import random
from collections import defaultdict

class GridWorld:

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.agent = [0, 0]
        self.goal = [self.height - 1, self.width - 1]
        self.wall = [1, 1]
        self.trap = [2, 2]
        self.reward = 0

    def step(self, direction):

        x, y = self.agent

        #move right
        if direction == 0:
            if y < self.width - 1 and [x, y + 1] != self.wall:
                y += 1

        #move left
        elif direction == 1:
            if y > 0 and [x, y - 1] != self.wall:
                y -= 1

        #move up
        elif direction == 2:
            if x < self.height - 1 and [x + 1, y] != self.wall:
                x += 1

        #move down
        elif direction == 3:
            if x > 0 and [x - 1, y] != self.wall:
                x -= 1

        self.agent = [x, y]
        done = (self.agent == self.goal)

        self.adjust_reward()

        return self.agent, self.reward, done

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
        grid = np.zeros((self.height, self.width))
        grid[self.agent[0], self.agent[1]] = 1
        grid[self.goal[0], self.goal[1]] = 2
        grid[self.wall[0], self.wall[1]] = 3
        grid[self.trap[0], self.trap[1]] = 4
        print(grid)

    def adjust_reward(self):
        if self.agent == self.trap:
            self.reward -= 1

        if self.agent == self.goal:
            self.reward += 1

        self.reward -= 0.1

def mc_estimation(env, n_epochs):    
    # Initialize empty dictionaries to store returns and counts for each state
    returns = np.zeros((env.width, env.height))
    count = np.zeros((env.width, env.height))

    # Run n_epochs epochs
    for epoch in range(n_epochs):
        # Generate an episode by following the agent's policy until the episode terminates
        episode = []
        state = env.reset()
        done = False

        while not done:
            decision = random.choices([0, 1], [0.2, 0.8])
            if decision == 1:
                direction = random.choice(env.evaluate())

            else:
                direction = random.choice([0, 1, 2, 3])

            next_state, reward, done = env.step(direction)
            episode.append((state, direction, reward))
            state = next_state

        # Update returns and counts for each state visited in the episode
        visited = np.zeros((env.width, env.height), dtype=bool)
        for t, (state, direction, reward) in enumerate(episode):
            x, y = state
            if not visited[x, y]:
                visited[x, y] = True
                G = sum(r for _, _, r in episode[t:])
                returns[x, y] += G
                count[x, y] += 1

    # Calculate the MC estimation for each state
    mc_estimates = np.round(returns / count, 2)

    return mc_estimates



if __name__ == '__main__':
    env = GridWorld(4, 4)

    mc_estimates = mc_estimation(env, n_epochs = 1)
    print(mc_estimates)

    # while True:

    #     #policy: move closer to the goal with a prob. of 0.8 else move random
    #     decision = random.choices([0, 1], [0.2, 0.8])

    #     if decision == 1:
    #         direction = random.choice(env.evaluate())

    #     else:
    #         direction = random.choice([0, 1, 2, 3])

    #     _, reward, done = env.step(direction)
    #     env.render()

    #     if done:
    #         break
    # print(env.reward)
