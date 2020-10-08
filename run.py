# -*- coding: utf-8 -*-

import numpy as np
import visualization as vis
from frozenlake_env import FrozenLake

MAP          = 0 # 0 for 4x4 and 1 for 10x10
NUM_EPISODE  = 1000
GAMMA        = 0.9
LR           = 0.1
EPSILON      = 0.1
ACTION_LIST  = [0, 1, 2, 3]
NUM_ACTION   = len(ACTION_LIST)
env          = FrozenLake(MAP)
if MAP == 0:
    MAP_X, MAP_Y = 4, 4
else:
    MAP_X, MAP_Y = 10, 10

def init_tables():
    """
    :return: Empty dictionary of policy, Q and return
    """
    policy_table, Q_table, return_table = {}, {}, {}
    for y in range(MAP_Y):
        for x in range(MAP_X):
            state = x, y
            policy_table[state] = [1 / NUM_ACTION] * NUM_ACTION
            Q_table[state] = [0] * NUM_ACTION
            return_table[state] = {}
            for a in ACTION_LIST:
                return_table[state][a] = []
    return policy_table, Q_table, return_table

def gen_episode(policy_table):
    done = False
    state = env.reset()
    state_list, action_list, reward_list, return_list = [], [], [], []
    while not done:
        action = np.random.choice(ACTION_LIST, p=policy_table[state])
        state_list.append(state)
        action_list.append(action)
        state, reward, done= env.step(action)
        reward_list.append(reward)
    G = 0
    for i in range(len(state_list)-1, -1, -1):
        G = GAMMA * G + reward_list[i]
        return_list.append(G)
    return_list.reverse()
    return state_list, action_list, return_list, reward_list

def find_rand_max_idx(Q_list):
    idx_list = []
    for idx, q in enumerate(Q_list):
        if q == max(Q_list):
            idx_list.append(idx)
    action = np.random.choice(idx_list)
    return action

def run_first_visit_MC():
    policy_table, Q_table, return_table = init_tables()
    for episode in range(NUM_EPISODE):
        state_list, action_list, return_list, reward_list = gen_episode(policy_table)
        _, _, return_table_temp = init_tables()
        for i in range(len(state_list)):
            state, action = state_list[i], action_list[i]
            if not return_table_temp[state][action]:
                return_table_temp[state][action].append(return_list[i])
                return_table[state][action].append(return_list[i])
            Q = np.mean(return_table[state][action])
            Q_table[state][action] = Q
            best_action = find_rand_max_idx(Q_table[state])
            for a in range(NUM_ACTION):
                if a == best_action:
                    policy_table[state][a] = 1 - EPSILON + EPSILON / NUM_ACTION
                else:
                    policy_table[state][a] = EPSILON / NUM_ACTION
        vis.show_progress(episode, NUM_EPISODE)
        if env.is_success(policy_table, episode):
            vis.show_success(episode)
            break
    print('First-visit Monte Carlo needs more training.')


def run_every_visit_MC():
    policy_table, Q_table, return_table = init_tables()
    for episode in range(NUM_EPISODE):
        state_list, action_list, return_list, reward_list = gen_episode(policy_table)
        for i in range(len(state_list)):
            state, action = state_list[i], action_list[i]
            return_table[state][action].append(return_list[i])
            Q = np.mean(return_table[state][action])
            Q_table[state][action] = Q
            best_action = find_rand_max_idx(Q_table[state])
            for a in range(NUM_ACTION):
                if a == best_action:
                    policy_table[state][a] = 1 - EPSILON + EPSILON / NUM_ACTION
                else:
                    policy_table[state][a] = EPSILON / NUM_ACTION
        vis.show_progress(episode, NUM_EPISODE)
        if env.is_success(policy_table, episode):
            vis.show_success(episode)
            break
    print('Every-visit Monte Carlo needs more training.')
    return policy_table

def run_SARSA():
    policy_table, Q_table, _ = init_tables()
    for episode in range(NUM_EPISODE):
        done = False
        state = env.reset()
        action = np.random.choice(ACTION_LIST, p=policy_table[state])
        while not done:
            new_state, reward, done = env.step(action)
            new_action = np.random.choice(ACTION_LIST, p=policy_table[new_state])
            new_Q = Q_table[new_state][new_action]
            Q_table[state][action] += LR * (reward + GAMMA * new_Q - Q_table[state][action])
            best_action = find_rand_max_idx(Q_table[state])
            for a in range(NUM_ACTION):
                if a == best_action:
                    policy_table[state][a] = 1 - EPSILON + EPSILON / NUM_ACTION
                else:
                    policy_table[state][a] = EPSILON / NUM_ACTION
            state = new_state
            action = new_action
        vis.show_progress(episode, NUM_EPISODE)
        if env.is_success(policy_table, episode):
            vis.show_success(episode)
            break
    print('SARSA needs more training.')
    return policy_table

def run_Q_learning():
    policy_table, Q_table, _ = init_tables()
    for episode in range(NUM_EPISODE):
        done = False
        state = env.reset()
        action = np.random.choice(ACTION_LIST, p=policy_table[state])
        while not done:
            new_state, reward, done = env.step(action)
            new_action = np.random.choice(ACTION_LIST, p=policy_table[new_state])
            new_Q = max([Q_table[new_state][a] for a in ACTION_LIST])
            Q_table[state][action] += LR * (reward + GAMMA * new_Q - Q_table[state][action])
            best_action = find_rand_max_idx(Q_table[state])
            for a in range(NUM_ACTION):
                if a == best_action:
                    policy_table[state][a] = 1 - EPSILON + EPSILON / NUM_ACTION
                else:
                    policy_table[state][a] = EPSILON / NUM_ACTION
            state = new_state
            action = new_action
        vis.show_progress(episode, NUM_EPISODE)
        if env.is_success(policy_table, episode):
            vis.show_success(episode)
            break
    print('Q learning needs more training.')
    return policy_table


if __name__ == '__main__':
    run_SARSA()
    run_Q_learning()
    run_first_visit_MC()
    run_every_visit_MC()





