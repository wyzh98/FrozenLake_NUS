# -*- coding: utf-8 -*-
"""
Created on Fri May  8 19:19:05 2020

@author: Tisana
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 20:59:36 2020

@author: Tisana
"""

import numpy as np
from racetrack_env import RacetrackEnv
import matplotlib.pyplot as plt
import random
env = RacetrackEnv()
gamma=0.9
alpha=0.01
epsilon=0.1
all_possible_action=[0, 1, 2, 3]
num_episode=100
to_avg=1
seed = 5
random.seed(seed)
np.random.seed(seed)
#random.seed(2)

#initialise tables
def initialise_tables():

    policy_table={} #key is state : value is list of prob of taking action a index by 0 to 8
    Q_table={} #key is state : value is list of Q value index by action 0 to 8
    return_table={} #key1 is state : key2 is action : value is return
    all_possible_action=[0, 1, 2, 3]
    num_y_position=4
    num_x_position=4
    for y_position in range(0,num_y_position):
        for x_position in range(0,num_x_position):
            state=(y_position,x_position)
            policy_table[state]=[1/len(all_possible_action)]*len(all_possible_action)
            Q_table[state]=[0]*len(all_possible_action)
            return_table[state]={}
            for each_action in all_possible_action:
               return_table[state][each_action]=[]
    return policy_table,Q_table,return_table
                

#play function taht play according to policy
#function output state,action,return list

def play_episode(policy_table):
    current_state=env.reset()
    state_list=[]
    reward_list=[]
    action_list=[]
    while True:
        action_to_take=np.random.choice(all_possible_action,p=policy_table[current_state])   
        state_list.append(current_state)
        action_list.append(action_to_take)
        current_state,reward,terminate=env.step(action_to_take)  
        reward_list.append(reward)
        if terminate:
            break 
    return_list=[]
    
    #calculate return list
    state_list.reverse()
    reward_list.reverse()
    action_list.reverse()  
    G=0
    for each_step in range(0,len(action_list)):
        G=(G*gamma)+reward_list[each_step]
        return_list.append(G)
    
    state_list.reverse()
    reward_list.reverse()
    action_list.reverse()
    return_list.reverse()       
    return   state_list,action_list,return_list,reward_list  



#return calculation list

############################################################################
############################################################################
############################################################################
##monte carlo


#initialise tables
policy_table,Q_table,return_table=initialise_tables()
#start loop for each episode
return_list_MC=[]
episode_list=[]
for each_episode in range(0,num_episode):
    #play for atleast 20 times
    return_value=0
    for j in range(0,to_avg):
        return_value+=sum(play_episode(policy_table)[3])
    return_value=return_value/ to_avg   
    return_list_MC.append(return_value)
    episode_list.append(each_episode)
    
    if each_episode%10==0:
        print("\r",each_episode,end="")
    state_list,action_list,return_list,_ =play_episode(policy_table)
    seen_state_list=[]
    for each_step in range(0,len(state_list)):
        state_action=[state_list[each_step],action_list[each_step]]
        
        #check for first visit condition
        if state_action not in seen_state_list:
            #append G into return list
            return_table[state_list[each_step]][action_list[each_step]].append(return_list[each_step])
            #update Q table
            avg_value=np.mean(return_table[state_list[each_step]][action_list[each_step]])
            Q_table[state_list[each_step]][action_list[each_step]]=avg_value
            #determine best action
            best_action=Q_table[state_list[each_step]].index(max(Q_table[state_list[each_step]]))
            
            for each_action in range(0,len(all_possible_action)):
                if each_action!=best_action:
                    prob=epsilon/len(all_possible_action)
                else:
                    prob=1-epsilon+(epsilon/len(all_possible_action))
                    
                #update policy table
                policy_table[state_list[each_step]][each_action]=prob
print(".")                
plt.figure("MC")
title="avg of "+str(to_avg)+" agents"
plt.title(title)
plt.xlabel("number of episodes")
plt.ylabel("mean return")
plt.plot(episode_list,return_list_MC)
plt.show()                
############################################################################
############################################################################
############################################################################
##sarsa

#initialise tables
policy_table,Q_table=initialise_tables()[:2]
return_list_sarsa=[]
episode_list=[]
#for each episode
for each_episode in range(0,num_episode):
    if each_episode%10==0:
        print("\r",each_episode,end="")
    #play for atleast 20 times
    return_value=0
    for j in range(0,to_avg):
        return_value+=sum(play_episode(policy_table)[3])
    return_value=return_value/ to_avg   
    return_list_sarsa.append(return_value)
    episode_list.append(each_episode)
    
    #initialise s
    current_state=env.reset()
    
    #choose action
    current_action=np.random.choice(all_possible_action,p=policy_table[current_state])   
    
    ##loop until s is terminal
    while True:
        #take current action for current state
        new_state,reward,terminate=env.step(current_action) 
        
        #choose new action for new state
        new_action=np.random.choice(all_possible_action,p=policy_table[new_state])   
        
        #update Q table
        Q_value_of_new_state_action=Q_table[new_state][new_action]
        Q_table[current_state][current_action]=Q_table[current_state][current_action]+(alpha*(reward+(gamma*Q_value_of_new_state_action)-Q_table[current_state][current_action]))
        
        #update policy table
        best_action_index=Q_table[current_state].index(max(Q_table[current_state]))
        prob_list=[]
        for i in range(0,len(all_possible_action)):
            if i==best_action_index:
                prob=1-epsilon+(epsilon/len(all_possible_action))
            else:
                prob=epsilon/len(all_possible_action)
            policy_table[current_state][i]=prob
        #set new state and action as current state and action
        current_state=new_state
        current_action=new_action
        
        #if terminal break
        if terminate:
            break

print(".")
plt.figure("SARSA")
title="avg of "+str(to_avg)+" agents"
plt.title(title)
plt.xlabel("number of episodes")
plt.ylabel("mean return")
plt.plot(episode_list,return_list_sarsa)
plt.show()
############################################################################
############################################################################
############################################################################
##Q learning

#initialise tables
policy_table,Q_table=initialise_tables()[:2]

return_list_Qlearning=[]
episode_list=[]
#for each episode
for each_episode in range(0,num_episode):
    if each_episode%10==0:
        print("\r",each_episode,end="")    
    #play for atleast 20 times
    return_value=0
    for j in range(0,to_avg):
        return_value+=sum(play_episode(policy_table)[3])
    return_value=return_value/ to_avg   
    return_list_Qlearning.append(return_value)
    episode_list.append(each_episode)
    
    #initialise s
    current_state=env.reset()
    
    #loop until terminal
    while True:
        #choose action
        action_to_take=np.random.choice(all_possible_action,p=policy_table[current_state])   
        
        #take action
        new_state,reward,terminate=env.step(action_to_take)
        
        #update Q table
        max_Q=max(Q_table[new_state])
        second_term=alpha*(reward+(gamma*max_Q)-Q_table[current_state][action_to_take])
        Q_table[current_state][action_to_take]=Q_table[current_state][action_to_take]+second_term
        
        #update policy table
        best_action_index=Q_table[current_state].index(max(Q_table[current_state]))
        prob_list=[]
        for i in range(0,len(all_possible_action)):
            if i==best_action_index:
                prob=1-epsilon+(epsilon/len(all_possible_action))
            else:
                prob=epsilon/len(all_possible_action)
            policy_table[current_state][i]=prob
            
        #set new state  as current state 
        current_state=new_state
        
        #if terminal break
        if terminate==True:
            break
print(".")
plt.figure("Q_learning")
title="avg of "+str(to_avg)+" agents"
plt.title(title)
plt.xlabel("number of episodes")
plt.ylabel("mean return")
plt.plot(episode_list,return_list_Qlearning)
plt.show()


############################################################################
############################################################################
############################################################################
#comparison

plt.figure("compare all 3 methods")
plt.xlabel("number of episodes")
plt.ylabel("mean return")
title="avg of "+str(to_avg)+" agents"
plt.title(title)
plt.plot(episode_list,return_list_MC,label="MC",c="red")
plt.plot(episode_list,return_list_sarsa,label="SARSA",c="blue")
plt.plot(episode_list,return_list_Qlearning,label="Q-learning",c="green")
plt.legend()
#plt.ylim(-10000,0)
plt.show()
############################################################################
############################################################################
############################################################################


