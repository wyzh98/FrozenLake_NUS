# -*- coding: utf-8 -*-

import numpy as np
from frozenlake_env import FrozenLake


class FindPathBase:

    def __init__(self, map_idx=1, num_episode=2000):
        self.name         = None
        self.MAP          = map_idx # 0 for 4x4 and 1 for 10x10
        self.NUM_EPISODE  = num_episode
        self.GAMMA        = 0.95
        self.EPSILON      = 0.1
        self.ACTION_LIST  = [0, 1, 2, 3]
        self.MAX_STEP     = 200
        self.NUM_ACTION   = len(self.ACTION_LIST)
        self.env          = FrozenLake(self.MAP)
        self.MAP_X, self.MAP_Y = (4, 4) if self.MAP == 0 else (10, 10)

    def init_tables(self):
        """
        :return: Empty dictionary of policy, Q and return
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

    def find_rand_max_idx(self, Q_list):
        idx_list = []
        for idx, q in enumerate(Q_list):
            if q == max(Q_list):
                idx_list.append(idx)
        action = np.random.choice(idx_list)
        return action


class FirstMonteCarlo(FindPathBase):

    def __init__(self, map_idx=1, num_episode=2000):
        super().__init__(map_idx=map_idx, num_episode=num_episode)
        self.name = 'Monte Carlo'

    def gen_episode(self, policy_table:dict):
        step = 0
        done = False
        state = self.env.reset()
        state_list, action_list, reward_list, return_list = [], [], [], []
        while not done:
            step += 1
            action = np.random.choice(self.ACTION_LIST, p=policy_table[state])
            state_list.append(state)
            action_list.append(action)
            state, reward, done= self.env.step(action)
            reward_list.append(reward)
            if step > self.MAX_STEP:
                break
        G = 0
        for i in range(len(state_list)-1, -1, -1):
            G = self.GAMMA * G + reward_list[i]
            return_list.append(G)
        return_list.reverse()
        return state_list, action_list, return_list

    def run(self):
        policy_table, Q_table, return_table = self.init_tables()
        for episode in range(self.NUM_EPISODE):
            state_list, action_list, return_list = self.gen_episode(policy_table)
            _, _, return_table_temp = self.init_tables()
            for i in range(len(state_list)):
                state, action = state_list[i], action_list[i]
                if not return_table_temp[state][action]:
                    return_table_temp[state][action].append(return_list[i])
                    return_table[state][action].append(return_list[i])
                Q = np.mean(return_table[state][action])
                Q_table[state][action] = Q
                best_action = self.find_rand_max_idx(Q_table[state])
                for a in range(self.NUM_ACTION):
                    if a == best_action:
                        policy_table[state][a] = 1 - self.EPSILON + self.EPSILON / self.NUM_ACTION
                    else:
                        policy_table[state][a] = self.EPSILON / self.NUM_ACTION
            self.env.render(len(action_list), episode, self.NUM_EPISODE, policy_table)
        self.env.render_all(self.NUM_EPISODE, self.name, policy_table, Q_table)
        print('First-visit Monte Carlo needs more training.')


class SARSA(FindPathBase):

    def __init__(self, map_idx=1, num_episode=2000):
        super().__init__(map_idx=map_idx, num_episode=num_episode)
        self.name = 'SARSA'
        self.LR   = 0.1

    def run(self):
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
                new_Q = Q_table[new_state][new_action]
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


class QLearning(FindPathBase):

    def __init__(self, map_idx=1, num_episode=2000):
        super().__init__(map_idx=map_idx, num_episode=num_episode)
        self.name = 'Q Learning'
        self.LR   = 0.1

    def run(self):
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
                new_Q = max([Q_table[new_state][a] for a in self.ACTION_LIST])
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
    map_idx = 1 # 0 for 4x4, 1 for 10x10
    env_MC = FirstMonteCarlo(map_idx, num_episode=2000)
    env_SARSA = SARSA(map_idx, num_episode=2000)
    env_QL = QLearning(map_idx, num_episode=2000)
    env_SARSA.run()




