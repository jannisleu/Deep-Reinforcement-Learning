import numpy as np
import random

class GridWorld:

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.agent = [0, 0]
        self.goal = [self.height - 1, self.width - 1]
        self.wall = [0, 1]
        self.trap = [2, 4]
        self.reward = 100

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

    def reset(self):
        self.agent = [0, 0]

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
            self.reward -= 10
        self.reward -= 0.1


if __name__ == '__main__':
    env = GridWorld(8, 8)

    while env.agent != env.goal:

        #policy: move closer to the goal with a prob. of 0.8 else move random
        decision = random.choices([0, 1], [0.2, 0.8])

        if decision == 1:
            direction = random.choice(env.evaluate())

        else:
            direction = random.choice([0, 1, 2, 3])

        env.step(direction)
        env.render()
        env.adjust_reward()

    print(env.reward)
