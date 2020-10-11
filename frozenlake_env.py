# -*- coding: utf-8 -*-

import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from more_itertools import chunked


class FrozenLake:

    def __init__(self, map_idx):

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
        self.map               = np.loadtxt(self.filepath[map_idx], dtype=int)
        self.MAP_X, self.MAP_Y = self.map.shape
        self.init_states       = [(x, y) for x in range(self.MAP_X) for y in range(self.MAP_Y) if self.CELL[self.map[x, y]] == 'start']
        self.e_fail            = []
        self.e_success         = []
        self.e_fail_len        = []
        self.e_success_len     = []
        self.e_opt_policy      = []

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
        self._show_progress(e)
        if self.CELL[self.map[self.state]] == 'frisbee':
            self.e_success.append(e)
            self.e_success_len.append(e_len)
        else:
            self.e_fail.append(e)
            self.e_fail_len.append(e_len)
        if self._is_success(policy):
            self.e_opt_policy.append(e)
        if e == e_all - 1:
            if self.e_success:
                print('Frisbee firstly reach at episode:', self.e_success[0])
            else:
                print('More training is needed.')

    def render_all(self, e_all, name, policy_table, Qtable):
        # plt.figure(1)
        l1 = plt.scatter(self.e_fail, self.e_fail_len, s=0.8, alpha=0.2, label='Fail')
        l2 = plt.scatter(self.e_success, self.e_success_len, s=0.8, c='red', alpha=1.0)
        l3 = plt.vlines(self.e_opt_policy, 0, max(self.e_fail_len), colors='green', alpha=0.02)
        plt.legend(handles=[l1,l2,l3], labels=['Fail','Success','Optimal policy'],loc='upper left')
        plt.title('Training Process of '+ name)
        plt.xlabel('#Episode')
        plt.ylabel('Step length')
        self.render_heatmap(policy_table, Qtable)
        self.render_sucess_rate()
        plt.show()

    def render_heatmap(self, policy_table, Qtable):
        num_cell = max(Qtable.keys())[0] + 1
        Q = np.full((num_cell, num_cell), 0, dtype=float)
        mask = np.full((num_cell, num_cell), False, dtype=bool)
        for x, y in Qtable.keys():
            Q[x][y] = max(Qtable[(x, y)])
        for y in range(self.MAP_Y):
            for x in range(self.MAP_X):
                if self.CELL[self.map[(x, y)]] == 'hole':
                    mask[x][y] = True
                elif self.CELL[self.map[(x, y)]] == 'frisbee':
                    mask[x][y] = True
        plt.figure(2)
        sns.heatmap(Q, annot=True, cmap='RdBu_r', square=True, mask=mask, linewidths=0.3, linecolor='black')
        plt.title('Max state-action value given state')

    def render_sucess_rate(self, smooth_size=20):
        e_smooth, s = [], []
        num_e = len(self.e_fail) + len(self.e_success)
        for batch in range(num_e // smooth_size):
            e_smooth.append(batch * smooth_size + smooth_size // 2)
            cnt = 0
            for e in range(batch*smooth_size, (batch + 1)*smooth_size):
                if e in self.e_success:
                    cnt += 1
            s.append(100 * cnt / smooth_size)
        plt.figure(3)
        plt.plot(e_smooth, s)
        plt.title('Success rate')
        return e_smooth, s

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

def smooth(L, smooth_size=20):
    return [sum(x) / len(x) for x in chunked(L, smooth_size)]

      