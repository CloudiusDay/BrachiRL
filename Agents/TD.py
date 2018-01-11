import numpy as np
from math import *
import random
import matplotlib.pyplot as plt
import math


#More work:
#1. Have a test mode, where the state-action value function is no longer updated. 
#2. The action is always greedy


class Agent:

    def __init__(self, delta_x = 0.5, delta_y = 0.05, range_x = (0,10), range_y = (0,10)):
        #The state space is discretized, the state is represented by (i,j), where i is the y axis and j is the x axis. 
        #action is represented by a, the action space at each state (i, j) is defined in self.set_A
        
        #Configure the Marte Carlo Agent:
        self.epsilon = 0.1  #For the epsilon greedy policy
        self.alpha = 0.2  #learning rate
        self.gamma = 1  #discount rate in calculating the 
        
        
        #In an actual application, the actor may not know the state space. Here, it's defined so that we can represent the action value function in a more compact and efficient way.              
        self.delta_x = delta_x
        self.delta_y = delta_y
        
        self.range_x = range_x
        self.range_y = range_y
        
        self.shape = ( round((self.range_y[1]-self.range_y[0])/self.delta_y) + 1, round((self.range_x[1]-self.range_x[0])/self.delta_x) + 1 )
        print('Shape of the state space: ', self.shape)
        
        #Initialize the action value function, which is the average of all returns seen so far, this is what is learnt and should not be intialized between episodes
        self.Q = [[{} for i in range(self.shape[1])] for i in range(self.shape[0])]        
        
        self.new_episode_reset()



    def new_episode_reset(self):
        #I also have to store the previous action, initialize this between episodes
        self.prev_action = None
        self.prev_state = None



    def act(self, state, reward = None, done = False, actionSet = None):  
        #Input state and reward, update action value function and return an action  



        
        #Find the greedy action with respect to the current state action value function
        greedy_action = None
        #Find action from self.Q
        value = math.inf
        for k, v in self.Q[state[0]][state[1]].items():
            if v < value:
                value = v
                greedy_action = k
            
        if greedy_action == None and actionSet:
            greedy_action = np.random.choice(actionSet)  

        
        #Find the self.epsilon soft greedy action:
        action = None  

        if actionSet:
            prob = 1-self.epsilon+self.epsilon/len(actionSet)
            #Add a bit of exploring start?
            if len(self.Q[state[0]][state[1]]) < 0.8*len(actionSet):
                prob = 0.5 
        else:
            prob = 1


        if np.random.uniform() <= prob:
            action = greedy_action  
        else:            
            action = np.random.choice(actionSet)
        
        #if action == None: print("No action is generated!")  #be careful here that you can't use "not action" like you test for empty list 
                  
        
        if state != None and action != None:
            i = state[0]
            j = state[1]
            #Q[i][j][action] is the current action-state value
            
            if action not in self.Q[i][j]:
                self.Q[i][j][action] = 0   #Initialize the action-state value to zero if it does not exist. Is this the best way?
                

        #Update the previous action-state value
        if (reward != None and
            self.prev_state != None and self.prev_action != None and
            state != None and action != None):

            i_prev = self.prev_state[0]
            j_prev = self.prev_state[1]
            a_prev = self.prev_action 
            #Q[i_prev][j_prev][a_prev] is previous action-state value 

            
            #previous action state must exist in self.Q(if it was not then it would be added in the previous step) 
            #but the current action state may not be in self.Q   
            self.Q[i_prev][j_prev][a_prev] = (1-self.alpha)*self.Q[i_prev][j_prev][a_prev] + self.alpha*(reward + self.gamma*self.Q[i][j][action])

        
        #Set the previous state and action for the next iteration:
        self.prev_state = state
        self.prev_action = action        


        if done: return None
   
        return action




    def actQ(self, state, reward = None, done = False, actionSet = None):  
        #Input state and reward, update action value function and return an action  



        
        #Find the greedy action with respect to the current state action value function
        greedy_action = None
        #Find action from self.Q
        value = math.inf
        for k, v in self.Q[state[0]][state[1]].items():
            if v < value:
                value = v
                greedy_action = k
            
        if greedy_action == None and actionSet:
            greedy_action = np.random.choice(actionSet)  

        
        #Find the self.epsilon soft greedy action:
        action = None  

        if actionSet:
            prob = 1-self.epsilon+self.epsilon/len(actionSet)
            #Add a bit of exploring start?
            if len(self.Q[state[0]][state[1]]) < 0.8*len(actionSet):
                prob = 0.5 
        else:
            prob = 1


        if np.random.uniform() <= prob:
            action = greedy_action  
        else:            
            action = np.random.choice(actionSet)
        
        #if action == None: print("No action is generated!")  #be careful here that you can't use "not action" like you test for empty list 
                  
        
        if state != None and greedy_action != None:
            i = state[0]
            j = state[1]
            #Q[i][j][action] is the current action-state value
            
            if greedy_action not in self.Q[i][j]:
                self.Q[i][j][greedy_action] = 0   #Initialize the action-state value to zero if it does not exist. Is this the best way?
                

        #Update the previous action-state value
        if (reward != None and
            self.prev_state != None and self.prev_action != None and
            state != None and greedy_action != None):

            i_prev = self.prev_state[0]
            j_prev = self.prev_state[1]
            a_prev = self.prev_action 
            #Q[i_prev][j_prev][a_prev] is previous action-state value 

            
            #previous action state must exist in self.Q(if it was not then it would be added in the previous step) 
            #but the current action state may not be in self.Q   
            #print(self.prev_state, self.prev_action)
            #print(state, action)            
            print(greedy_action)
            print(a_prev)
            self.Q[i_prev][j_prev][a_prev] = (1-self.alpha)*self.Q[i_prev][j_prev][a_prev] + self.alpha*(reward + self.gamma*self.Q[i][j][greedy_action])

        
        #Set the previous state and action for the next iteration:
        self.prev_state = state
        self.prev_action = action        


        if done: return None
   
        return action



    
