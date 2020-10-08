# -*- coding: utf-8 -*-

import random
import numpy as np
import visualization as vis
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from matplotlib import colors

class FrozenLake:

    def __init__(self, map_version):

        self.ACTIONS = {
            0: (-1, 0),  # Up,
            1: (0, 1),   # Right
            2: (1, 0),   # Down
            3: (0, -1),  # Left
        }

        self.CELL = {
            0: 'ice',
            1: 'hole',
            2: 'start',
            3: 'frisbee',
        }
        
        self.filepath = ['./track_4x4.txt', './track_10x10.txt']
        self.state             = None
        self.map               = np.loadtxt(self.filepath[map_version], dtype=int)
        self.MAP_X, self.MAP_Y = self.map.shape
        self.init_states       = [(x, y) for x in range(self.MAP_X) for y in range(self.MAP_Y) if self.CELL[self.map[x, y]] == 'start']

    def reset(self):
        self.state = random.choice(self.init_states)
        return self.state

    def step(self, action:int):
        done    = False
        reward  = 0
        dx, dy  = self.ACTIONS[action]
        new_state = self.state[0] + dx, self.state[1] + dy

        if not (0 <= new_state[0] < self.MAP_X and 0 <= new_state[1] < self.MAP_Y):
            reward -= 0
        elif self.CELL[self.map[new_state]] in ['ice', 'start']:
            self.state = new_state
            reward -= 0
        elif self.CELL[self.map[new_state]] == 'hole':
            self.state = new_state
            reward -= 1
            done = True
        elif self.CELL[self.map[new_state]] == 'frisbee':
            self.state = new_state
            reward += 1
            done = True
        else:
            raise RuntimeWarning('No corresponding cell. Check your map file.')

        return self.state, reward, done

    def is_success(self, policy_table, episode):
        state = self.reset()
        done = False
        state_list = []
        while not done:
            action = int(np.argmax(policy_table[state]))
            dx, dy = self.ACTIONS[action]
            state_list.append(state)
            new_state = state[0] + dx, state[1] + dy
            if not (0 <= new_state[0] < self.MAP_X and 0 <= new_state[1] < self.MAP_Y):
                done = True
            elif new_state in state_list:
                done = True
            elif self.CELL[self.map[new_state]] == 'ice':
                state = new_state
            elif self.CELL[self.map[new_state]] in ['hole', 'start']:
                done = True
            elif self.CELL[self.map[new_state]] == 'frisbee':
                return True
        return False