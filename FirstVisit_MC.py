import matplotlib.pyplot as plt
import numpy as np
import seaborn
from FrozenLake_env import GridWorld

def generate_random_episode(env):
    episode = []
    done = False
    current_state = 0
    episode.append((current_state, 0))
    while not done:
        action = np.random.choice(env.actions)
        next_state, reward = env.state_transition(current_state, action)
        episode.append((next_state, reward))
        if current_state in [5, 7, 11, 12, 15]:
            done = True
        current_state = next_state
        # print('Running')
    return episode

def first_visit_mc(env, num_iter):
    FirstVisitValues = []
    FirstVisitIterations = []
    values = np.zeros(len(env.states) + 2)
    returns = dict()
    for state in env.states:
        returns[state] = list()

    for i in range(num_iter):
        episode = generate_random_episode(env)
        already_visited = {0}  # also exclude terminal state (0)
        for s, r in episode:
            if s == 15:
                break
            if s not in already_visited:
                already_visited.add(s)
                idx = episode.index((s, r))
                G = 0
                j = 1
                while j + idx < len(episode):
                    G = env.gamma * (G + episode[j + idx][1])
                    FirstVisitValues.append(G)
                    FirstVisitIterations.append(j)
                    j += 1
                returns[s].append(G)
                values[s] = np.mean(returns[s])
    return values, returns, FirstVisitValues, FirstVisitIterations

def show_values(values, whichVisit):
    values = values.reshape(4, 4)
    ax = seaborn.heatmap(values, cmap="Blues_r", annot=True, linecolor="#282828", linewidths=0.1)
    if whichVisit == 1:
       plt.title('First Visit State Value Table')
       # plt.savefig('FirstVisitStateValueTable.png')
    elif whichVisit == 2:
        plt.title('Every Visit State Value Table')
        # plt.savefig('EveryVisitStateValueTable.png')
    plt.show()



def main():
    gw = GridWorld(gamma=0.9, theta=0.5)
    values, returns, FirstVisitValues, FirstVisitIterations = first_visit_mc(gw, 1000)
    show_values(values, 1)
    plt.plot(FirstVisitValues, FirstVisitIterations)
    plt.xlabel('Values in States')
    plt.ylabel('All States')
    plt.title('Values VS States(First Visit)')
    # plt.savefig('Values VS States(First Visit).png')
    plt.show()


if __name__ == '__main__':
    main()

