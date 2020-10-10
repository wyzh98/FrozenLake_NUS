# -*- coding: utf-8 -*-

import random
import numpy as np
import matplotlib.pyplot as plt
from more_itertools import chunked


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
        self.e_success         = []
        self.e_len             = []

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

    def render(self, e_len, e, e_all, policy):
        self.e_len.append(e_len)
        self._show_progress(e)
        if self._is_success(policy):
            self.e_success.append(e)
        if e == e_all - 1:
            if self.e_success:
                print('Optimal policy generated at episode:', self.e_success[0])
            else:
                print('SARSA needs more training.')

    def render_all(self, e_all, policy, Qtable):
        E = list(range(1, e_all + 1))
        e_success_len = [self.e_len[i] for i in self.e_success]
        plt.figure(1)
        plt.scatter(E, self.e_len, s=0.8, alpha=1.0)
        plt.scatter(self.e_success, e_success_len, s=0.8, c='red', alpha=1.0)
        plt.show()

    def _is_success(self, policy_table):
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

    def _show_progress(self, e):
        if e % 10 == 0:
            print('Episode: %d' % e, end='\r')

def smooth( L, smooth_size=20):
    return [sum(x) / len(x) for x in chunked(L, smooth_size)]

      