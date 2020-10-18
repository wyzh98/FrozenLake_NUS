# -*- coding: utf-8 -*-
"""
Created on Sun Oct 4

@author: Wang Yizhuo
"""

import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class FrozenLake:
    """Environment of frozen lake.
    
    Attributes:
        map_idx: 0 for 4x4 map, 1 for 10x10 map.
    """    

    def __init__(self, map_idx):
        """Define and decode the map.
        Do not change the params here.

        Args:
            map_idx (int): map index
        """        

        self.ACTIONS = {  # coord in the matrix
            0: (-1,  0),  # Up
            1: ( 0,  1),  # Right
            2: ( 1,  0),  # Down
            3: ( 0, -1),  # Left
        }

        self.CELL = {     # interpret the map file
            0: 'ice',
            1: 'hole',
            2: 'start',
            3: 'frisbee',
        }
        
        self.filepath = ['./map_4x4.txt', './map_10x10.txt']
        self.state             = None
        self.map               = np.loadtxt(self.filepath[map_idx], dtype=int)
        self.MAP_X, self.MAP_Y = self.map.shape
        self.init_states       = [(x, y) for x in range(self.MAP_X) for y in range(self.MAP_Y) if self.CELL[self.map[x, y]] == 'start'] # find init state
        self.e_fail, self.e_success, self.e_opt_policy = {}, {}, []

    def reset(self):
        """Reset the state to the start

        Returns:
            tuple: state coord
        """   

        self.state = random.choice(self.init_states)
        return self.state

    def step(self, action:int):
        """Take a step given state and action, and return the next step and reward.

        Args:
            action (int): input an action

        Raises:
            RuntimeWarning: If there is no corresponding cell, the map file might contain illegal characters.

        Returns:
            tuple: next valid state given action
            float: the reward of next state
            bool: if next state is terminal
        """    

        done    = False
        reward  = 0
        dx, dy  = self.ACTIONS[action]
        new_state = self.state[0] + dx, self.state[1] + dy

        reward -= 0                                                # penalty for moving each timestep, 0 by default
        if not (0 <= new_state[0] < self.MAP_X and 0 <= new_state[1] < self.MAP_Y):
            reward -= 0                                            # penalty for going out of bound, 0 by default. Keep where they are
        elif self.CELL[self.map[new_state]] in ['ice', 'start']:
            self.state = new_state
            reward -= 0                                            # penalty for moving, 0 by default
        elif self.CELL[self.map[new_state]] == 'hole':
            self.state = new_state
            reward -= 1                                            # penalty for fall into the hole, -1 by default. terminal state
            done = True
        elif self.CELL[self.map[new_state]] == 'frisbee':
            self.state = new_state
            reward += 1                                            # reward for get the frisbee, +1 by default. terminalstate
            done = True
        else:
            raise RuntimeWarning('No corresponding cell. Check your map file.')

        return self.state, reward, done

    def render(self, e_len, e, e_all, policy):
        """Store the info for each episode.
        Detect whether the episode ends with getting the frisbee, and store it in the attributes.

        Args:
            e_len (int): length of the episode
            e (int): current episode
            e_all (int): number of total episode
            policy (dict): the policy table
        """   

        self._show_progress(e)                           # show progress of training on the sreen
        if self.CELL[self.map[self.state]] == 'frisbee': # store if the episode ends with getting the frisbee or not
            self.e_success[e] = e_len
        else:
            self.e_fail[e] = e_len
        if self._is_success(policy):                     # judge if the policy is optimal or not, record if it is optimal
            self.e_opt_policy.append(e)
        if e == e_all - 1:                               # judge at the last episode, show if a successful policy has been reached during the training
            if self.e_success:
                print('Frisbee firstly reach at episode:', list(self.e_success.keys())[0])
            else:
                print('More training is needed.')
            if e in self.e_success:
                self.render_policy(policy)               # show the policy

    def render_all(self, e_all, name, policy_table, Qtable):
        """Render graphs when finished the training.

        Args:
            e_all (int): number of total episode
            name (string): name of the algorithm
            policy_table (dict): policy table
            Qtable (dict): Q table
        """        

        plt.figure()
        l1 = plt.scatter(self.e_fail.keys(), self.e_fail.values(), s=0.8, alpha=0.3, label='Fail')     # plot failed points in blue
        l2 = plt.scatter(self.e_success.keys(), self.e_success.values(), s=0.8, c='red', alpha=1.0)    # plot successful points in red
        l3 = plt.vlines(self.e_opt_policy, 0, max(self.e_fail.values()), colors='green', alpha=0.05)   # plot optimal policy lines in green
        plt.legend(handles=[l1,l2,l3], labels=['Fail','Success','Optimal policy'], loc='upper right')
        plt.title('Training Process of '+ name)
        plt.xlabel('#Episode')
        plt.ylabel('Step length')
        # plt.savefig(r'xxx.png', dpi=300)                # save figure for report
        self.render_heatmap(policy_table, Qtable, name)   # render the heatmap

    def render_heatmap(self, policy_table, Qtable, name):
        """Render heatmap.

        Args:
            policy_table (dict): policy table
            Qtable (dict): Q table
            name (string): name of the algorithm
        """   

        num_cell = max(Qtable.keys())[0] + 1                    # number of the states
        Q = np.full((num_cell, num_cell), 0, dtype=float)       # create a square state matrix
        mask = np.full((num_cell, num_cell), False, dtype=bool) # for the use of mask of heatmap
        for x, y in Qtable.keys():
            Q[x][y] = max(Qtable[(x, y)])                       # put the values of Q table dict into the 2D array correspondingly
        for y in range(self.MAP_Y):
            for x in range(self.MAP_X):
                if self.CELL[self.map[(x, y)]] == 'hole':       # create mask, to make the cells at holes and target blank
                    mask[x][y] = True
                elif self.CELL[self.map[(x, y)]] == 'frisbee':
                    mask[x][y] = True
        plt.figure()
        sns.heatmap(Q, annot=True, cmap='RdBu_r', square=True, mask=mask, linewidths=0.3, linecolor='black', annot_kws={'size': 5}) # plot the heatmap with mask
        # sns.heatmap(Q, cmap='RdBu_r', square=True, mask=mask, linewidths=0.3, linecolor='black')   # heatmap without annotations
        plt.title('Max state-action value given state ({})'.format(name))
        # plt.savefig(r'{}.png'.format(name), dpi=300)                                               # save figure for report

    def render_policy(self, policy_table):
        """Show policy on the screen."""      

        done = False
        self.state = self.reset()
        state_list, action_list = [self.state], []
        while not done:
            action = int(np.argmax(policy_table[self.state]))
            _, _, done = self.step(action)
            state_list.append(self.state)
            action_list.append(action)
        print('Policy (state): {}'.format(state_list))
        print('Policy (action): {}'.format(action_list))

    def _is_success(self, policy_table):
        """Judge whether the policy is optimal.

        Args:
            policy_table (dict)

        Returns:
            bool: True if optimal, false otherwise
        """      

        state = self.reset()
        done = False
        state_list = []
        while not done:
            action = int(np.argmax(policy_table[state]))               # use argmax to extract the policy
            dx, dy = self.ACTIONS[action]
            state_list.append(state)
            new_state = state[0] + dx, state[1] + dy
            if not (0 <= new_state[0] < self.MAP_X and 0 <= new_state[1] < self.MAP_Y):
                done = True                                            # an optimal policy should not hit the wall
            elif new_state in state_list:
                done = True                                            # an optimal policy should not be recurrent
            elif self.CELL[self.map[new_state]] == 'ice':
                state = new_state
            elif self.CELL[self.map[new_state]] in ['hole', 'start']:
                done = True                                            # an optimal policy should not fall into the hole or back to the start
            elif self.CELL[self.map[new_state]] == 'frisbee':
                return True                                            # an optimal policy should lead to the target successfully
        return False

    def _show_progress(self, e):
        """Show progress on the sreen."""

        if e % 10 == 0:
            print('Episode: %d' % e, end='\r')

def render_learn_curve(envS, envQ, envM):
    """Render the learning curve of three algorithms.

    Args:
        envS (class): SARSA env
        envQ (class): Q-learning env
        envM (class): Monte Carlo env
    """    

    l     = []
    num_e = len(envS.env.e_fail) + len(envS.env.e_success)
    E     = list(range(num_e))
    plt.figure()
    for env in [envS, envQ, envM]:                 # loop for all the envs
        s_all = 0                                  # accumulative step number
        s     = []                                 # accumulative step number for each episode
        for e in range(num_e):
            if e in env.env.e_fail.keys():
                s_all += env.env.e_fail[e]
                s.append(s_all)
            elif e in env.env.e_success.keys():
                s_all += env.env.e_success[e]
                s.append(s_all)
        handle, = plt.plot(E, s)                   # comma after handle to unzip
        l.append(handle)
    plt.legend(handles=l, labels=[envS.name, envQ.name, envM.name])
    plt.xlabel('#Episode')
    plt.ylabel('Accumulative steps')
    plt.title('Comparison of learning curve')

def render_success_rate(envS, envQ, envM, smooth_size=20):
    """ Render success rate for each env.

    Args:
        envS (class): SARSA env
        envQ (class): Q-learning env
        envM (class): Monte Carlo env
        smooth_size (int, optional): take an average to a batch of episodes to smooth the curve. Defaults to 20.
    """    

    l = []
    num_e = len(envS.env.e_fail) + len(envS.env.e_success)
    plt.figure()
    for env in [envS, envQ, envM]:
        e_smooth, s = [], []
        for batch in range(num_e // smooth_size):
            e_smooth.append(batch * smooth_size + smooth_size // 2)     # center loc of the batch
            cnt = 0                                                     # counter for sccessful exploration
            for e in range(batch*smooth_size, (batch + 1)*smooth_size):
                if e in env.env.e_success.keys():
                    cnt += 1
            s.append(100 * cnt / smooth_size)                           # percentage of average sccess rate in a batch
        handle, = plt.plot(e_smooth, s)
        l.append(handle)
    plt.legend(handles=l, labels=[envS.name, envQ.name, envM.name])
    plt.title('Comparison of three method with a smoothness of {} episodes'.format(smooth_size))
