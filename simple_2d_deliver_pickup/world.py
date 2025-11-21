import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum

# Actions available to the simulated "robot"
class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    PICK_UP = 4
    DROP = 5


class GridWorld:
    def __init__(self, size=10, visualize = False):
        self.size = size
        self.visualize = visualize
        self.reset()
    
    def reset(self):
        self.agent_pos = np.random.randint(0, self.size, size=2)
        self.star_pos = np.random.randint(0, self.size, size=2)
        self.drop_pos = np.random.randint(0, self.size, size=2)
        self.has_star = 0
        self.steps = 0
        return
    
    def get_state(self):
        ax, ay = self.agent_pos
        sx, sy = self.star_pos
        dx, dy = self.drop_pos
        return np.array([ax, ay, sx, sy, dx, dy, self.has_star], dtype=np.float32)

    def step(self, action: Action):
        ax, ay = self.agent_pos

        # Movement
        if action == Action.UP:
            ay = max(0, ay - 1)
        elif action == Action.DOWN:
            ay = min(self.size - 1, ay + 1)
        elif action == Action.LEFT:
            ax = max(0, ax - 1)
        elif action == Action.RIGHT:
            ax = min(self.size - 1, ax + 1)
        elif action == Action.PICK_UP:
            if np.array_equal(self.agent_pos, self.star_pos):
                self.has_star = 1
        elif action == Action.DROP:
            if np.array_equal(self.agent_pos, self.drop_pos) and self.has_star:
                self.has_star = 0
                done = True
                return self.get_state(), 10.0, done  # success

        self.agent_pos = np.array([ax, ay])
        self.steps += 1

        reward = -0.1  # step penalty

        # If more then 100 steps are taken, be "done" and have negative reward
        done = self.steps >= 100
        return self.get_state(), reward, done
    

    def render(self, ax=None):
        if not self.visualize:
            return

        grid = np.zeros((self.size, self.size, 3))
        grid[self.star_pos[1], self.star_pos[0]] = [1, 1, 0]  # star (yellow)
        grid[self.drop_pos[1], self.drop_pos[0]] = [0, 0, 1]  # drop (blue)
        color = [1, 0, 0] if self.has_star == 0 else [0, 1, 0]
        grid[self.agent_pos[1], self.agent_pos[0]] = color

        if ax is None:
            ax = plt.gca()

        ax.clear()
        ax.imshow(grid)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.pause(0.2)
