# -*- coding: utf-8 -*-
"""
Created on Sun Oct 4

@author: Wang Yizhuo
"""

import numpy as np
import matplotlib.pyplot as plt
from frozenlake_env import FrozenLake, render_success_rate, render_learn_curve


class FindPathBase:
    """
    This is the base class of path finding, containing basical parameters setting and initialization.

    Attributes:
        map_idx: 0 for 4x4 map, 1 for 10x10 map.
        num_episode: number of training episode
    """

    def __init__(self, map_idx=1, num_episode=2000):
        """You can set your params here.

        Args:
            map_idx (int, optional): Defaults to 1.
            num_episode (int, optional): Defaults to 2000.
        """

        self.name         = None                                            # name for the class
        self.MAP          = map_idx                                         # 0 for 4x4 and 1 for 10x10
        self.NUM_EPISODE  = num_episode                                     # number of episode
        self.GAMMA        = 0.95                                            # discount rate
        self.EPSILON      = 0.1                                             # episilon greedy
        self.ACTION_LIST  = [0, 1, 2, 3]
        self.MAX_STEP     = np.inf                                          # maximum step per episode, infinite by default
        self.NUM_ACTION   = len(self.ACTION_LIST)
        self.env          = FrozenLake(self.MAP)
        self.MAP_X, self.MAP_Y = (4, 4) if self.MAP == 0 else (10, 10)

    def init_tables(self):
        """Initialize tables with certain data structure.

        Returns:
            dict: (policy_table) A dict mapping every state to action prob list. 
                Note: sum of every action list is 1, init with avg.
                For example: {(0, 0): [0.25, 0.25, 0.25, 0.25], ...}

            dict: (Q_table) A dict mapping every state to action score list. Init with 0.
                For example: {(0, 0): [0, 0, 0, 0], ...}

            dict: (return_table) A dict mapping every state to another dict that mapping every action to a list.
                The list will further store return of every episode.
                For example: {(0, 0): {0: [], 1: [], 2: [], 3: []}, ...}
        """

        policy_table, Q_table, return_table = {}, {}, {}
        for y in range(self.MAP_Y):
            for x in range(self.MAP_X):
                state = x, y
                policy_table[state] = [1 / self.NUM_ACTION] * self.NUM_ACTION
                Q_table[state] = [0] * self.NUM_ACTION
                return_table[state] = {}
                for a in self.ACTION_LIST:
                    return_table[state][a] = []
        return policy_table, Q_table, return_table

    def find_rand_max_idx(self, Q_list) -> int:
        """Find the max value index of a list. 
        If there are many of them, randomly pick one.

        Args:
            Q_list (list): List that is not empty.

        Returns:
            int: The index of the list is the corresponding action.
        """   

        idx_list = []
        for idx, q in enumerate(Q_list):
            if q == max(Q_list):
                idx_list.append(idx)
        action = np.random.choice(idx_list)
        return action


class FirstMonteCarlo(FindPathBase):
    """First-visit Monte Carlo

    Args:
        FindPathBase (class): Base class.
    """    

    def __init__(self, map_idx=1, num_episode=2000):
        super().__init__(map_idx=map_idx, num_episode=num_episode)
        self.name = 'Monte Carlo'

    def gen_episode(self, policy_table:dict):
        """Generate a whole episode
        Start with an init position, and play the episodes while record the info till termination.

        Args:
            policy_table (dict): Policy table, used to choose action.

        Returns:
            list: (state_list) Every state tuple for an episode.
            list: (action_list) Every action number taken.
            list: (return_list) Every return for each state-action pair.
        """       

        step  = 0                                                               # record the step length
        done  = False
        state = self.env.reset()
        state_list, action_list, reward_list, return_list = [], [], [], []
        while not done:
            step += 1
            action = np.random.choice(self.ACTION_LIST, p=policy_table[state])  # choose an action according to p.d.f of policy table
            state_list.append(state)
            action_list.append(action)
            state, reward, done = self.env.step(action)
            reward_list.append(reward)
            if step > self.MAX_STEP:                 # break the loop if it is over the max step length
                break
        G = 0                                        # accumulative return
        for i in range(len(state_list)-1, -1, -1):   # trace back the episode
            G = self.GAMMA * G + reward_list[i]      # calculate the return by reward of each state
            return_list.append(G)
        return_list.reverse()                        # reverse the traced back list to positive order
        return state_list, action_list, return_list

    def run(self):
        """Core function.
        Update the policy table, Q table and return table.

        Returns:
            dict: Policy table.
        """        

        policy_table, Q_table, return_table = self.init_tables()                   # init the tables
        for episode in range(self.NUM_EPISODE):                                    # loop for each episode
            state_list, action_list, return_list = self.gen_episode(policy_table)  # generate a whole episode and return three lists
            _, _, return_table_temp = self.init_tables()                           # make a empty temp return table, to judge if first-visit. clean for each episode
            for i in range(len(state_list)):                                       # loop to update the values in three tables
                state, action = state_list[i], action_list[i]
                if not return_table_temp[state][action]:                           # judge if it is first-visit
                    return_table_temp[state][action].append(return_list[i])        # if so, mark in the temp list and append to the return list
                    return_table[state][action].append(return_list[i])
                Q = np.mean(return_table[state][action])                           # calculate the average return and update the Q table
                Q_table[state][action] = Q
                best_action = self.find_rand_max_idx(Q_table[state])               # choose best action by argmax(Q(s, a))
                for a in range(self.NUM_ACTION):
                    if a == best_action:                                           # episilon-greedy policy: exploitation
                        policy_table[state][a] = 1 - self.EPSILON + self.EPSILON / self.NUM_ACTION
                    else:
                        policy_table[state][a] = self.EPSILON / self.NUM_ACTION    # exploration
            self.env.render(len(action_list), episode, self.NUM_EPISODE, policy_table)  # render and store the info of this episode
        self.env.render_all(self.NUM_EPISODE, self.name, policy_table, Q_table)    # render all the info once finished training
        return policy_table


class SARSA(FindPathBase):
    """SARSA

    Args:
        FindPathBase (class): Base class.
    """    

    def __init__(self, map_idx=1, num_episode=2000):
        super().__init__(map_idx=map_idx, num_episode=num_episode)
        self.name = 'SARSA'
        self.LR   = 0.1                                     # learning rate declared here

    def run(self):
        """Core function of SARSA.

        Returns:
            dict: Policy table.
        """        

        policy_table, Q_table, _ = self.init_tables()       # SARSA do not need return table, since it is TD
        for episode in range(self.NUM_EPISODE):             # loop for each episode
            # self.EPSILON = 0.05 + 0.95/(0.1*episode+1)    # uncomment this line if want to try decaying epsilon
            # self.LR = 0.05 + 0.95/(episode+1)             # uncomment this line if want to try decaying learning rate
            step   = 0                                      # record step length per episode
            done   = False
            state  = self.env.reset()
            action = np.random.choice(self.ACTION_LIST, p=policy_table[state])                             # choose init action from policy p.d.f
            while not done:
                step += 1
                new_state, reward, done = self.env.step(action)                                            # take a step
                new_action = np.random.choice(self.ACTION_LIST, p=policy_table[new_state])                 # choose next action from policy p.d.f
                new_Q = Q_table[new_state][new_action]                                                     # calculate Q(s', a')
                Q_table[state][action] += self.LR * (reward + self.GAMMA * new_Q - Q_table[state][action]) # TD update rule
                best_action = self.find_rand_max_idx(Q_table[state])                                       # find best action by argmaxQ
                for a in range(self.NUM_ACTION):
                    if a == best_action:
                        policy_table[state][a] = 1 - self.EPSILON + self.EPSILON / self.NUM_ACTION         # epsilon-greedy
                    else:
                        policy_table[state][a] = self.EPSILON / self.NUM_ACTION
                state = new_state
                action = new_action
                if step > self.MAX_STEP:                    # break if over max step
                    break
            self.env.render(step, episode, self.NUM_EPISODE, policy_table)
        self.env.render_all(self.NUM_EPISODE, self.name, policy_table, Q_table)
        return policy_table


class QLearning(FindPathBase):
    """Q-learning

    Args:
        FindPathBase (class): Base class.
    """    

    def __init__(self, map_idx=1, num_episode=2000):
        super().__init__(map_idx=map_idx, num_episode=num_episode)
        self.name = 'Q-learning'
        self.LR   = 0.1                               #  learning rate declared here

    def run(self):
        """Core function of Q-learning.

        Returns:
            dict: Policy table.
        """     

        policy_table, Q_table, _ = self.init_tables()
        for episode in range(self.NUM_EPISODE):
            step   = 0
            done   = False
            state  = self.env.reset()
            action = np.random.choice(self.ACTION_LIST, p=policy_table[state])
            while not done:
                step += 1
                new_state, reward, done = self.env.step(action)
                new_action = np.random.choice(self.ACTION_LIST, p=policy_table[new_state])
                new_Q = max([Q_table[new_state][a] for a in self.ACTION_LIST])              # only difference with SARSA, use max of Q(s', a') as target policy
                Q_table[state][action] += self.LR * (reward + self.GAMMA * new_Q - Q_table[state][action])
                best_action = self.find_rand_max_idx(Q_table[state])
                for a in range(self.NUM_ACTION):
                    if a == best_action:
                        policy_table[state][a] = 1 - self.EPSILON + self.EPSILON / self.NUM_ACTION
                    else:
                        policy_table[state][a] = self.EPSILON / self.NUM_ACTION
                state = new_state
                action = new_action
                if step > self.MAX_STEP:
                    break
            self.env.render(step, episode, self.NUM_EPISODE, policy_table)
        self.env.render_all(self.NUM_EPISODE, self.name, policy_table, Q_table)
        return policy_table


if __name__ == '__main__':        
    map_idx     = 1                                          # map index: 0 for 4x4, 1 for 10x10
    n           = 2000                                       # number of episode
    env_MC      = FirstMonteCarlo(map_idx, num_episode=n)    # instantiation of Monte Carlo, with a total episode of n, same below
    env_SARSA   = SARSA(map_idx, num_episode=n)
    env_QL      = QLearning(map_idx, num_episode=n)
    env_SARSA.run()    # run SARSA, Q-learning, Monte Carlo in sequence
    env_QL.run()
    env_MC.run()
    render_success_rate(envS=env_SARSA, envQ=env_QL, envM=env_MC, smooth_size=20) # render success rate of three methods, comment if you don't run all three methods
    render_learn_curve(envS=env_SARSA, envQ=env_QL, envM=env_MC)                  # render learning curve of three methods
    # print('avg step', (sum(env_SARSA.env.e_success.values())+sum(env_SARSA.env.e_fail.values()))/2000) # uncomment if you want to know avg step of SARSA
    # print('max step:', max(max(env_SARSA.env.e_success.values()), max(env_SARSA.env.e_fail.values()))) # uncomment if you want to know max step of SARSA
    plt.show()         # show all the graphs

