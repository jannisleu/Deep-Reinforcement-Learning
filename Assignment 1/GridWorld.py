import numpy as np

class GridWorld:

    def __init__(self, height, width):
        self.height = height
        self.width = width
        #self.grid = np.array((self.height, self. width))
        self.agent = [0, 0]
        self.goal = [self.height - 1, self.width - 1]

    def step(self, direction):

        x, y = self.agent

        #move right
        if direction == 0:
            if y < self.width:
                y += 1

        #move left
        if direction == 1:
            if y > 0:
                y -= 1

        #move up
        if direction == 2:
            if x < self.height:
                x += 1

        #move down
        if direction == 3:
            if x > 0:
                x -= 1

        self.agent = [x, y]

    def reset(self):
        self.agent = [0, 0]

    def render(self):
        grid = np.zeros((self.height, self.width))
        grid[self.agent[0], self.agent[1]] = 1
        grid[self.goal[0], self.goal[1]] = 2
        print(grid)


if __name__ == '__main__':
    env = GridWorld(8, 8)
    env.step(0)
    env.render()
