import numpy as np
from math import *
import random
import matplotlib.pyplot as plt
import math



class Agent:

    def __init__(self, delta_x = 0.5, delta_y = 0.05, range_x = (0,10), range_y = (0,10)):
        #The state space is discretized, the state is represented by (i,j), where i is the y axis and j is the x axis. 
        #action is represented by a, the action space at each state (i, j) is defined in self.set_A
        
        #Configure the Q-learning agent:
        self.epsilon = 0.1  #For the epsilon greedy policy
        self.alpha = 0.2  #learning rate
        self.gamma = 1  #discount rate in calculating the return
        
        
        #This discretization should be part of the agent but not part of the environment     
        self.delta_x = delta_x
        self.delta_y = delta_y
        
        self.range_x = range_x
        self.range_y = range_y
        
        self.shape = ( round((self.range_y[1]-self.range_y[0])/self.delta_y) + 1, round((self.range_x[1]-self.range_x[0])/self.delta_x) + 1 )
        print('Shape of the state space: ', self.shape)
        
        #Initialize the action value function, which is the average of all returns seen so far, this is what is learnt and should not be intialized between episodes
        self.Q = [[{} for i in range(self.shape[1])] for i in range(self.shape[0])]        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):            
                for next_y_index in range(self.shape[0]):
                    self.Q[i][j][next_y_index - i] = random.random()       
        #Use this Q to generate an action, if this action is not in the current actionSet then find something random in the actionSet

        self.new_episode_reset()



    def new_episode_reset(self):
        self.prev_action = None
        self.prev_state = None




    def act(self, state, done, actionSet, reward = None):  
        #Input state and reward, update action value function and return an action  
        
        if state == None:
            raise Exception('There is no observation to make a decision.')

        if done: return None
               

        #Find the greedy action with respect to the current state action value function
        greedy_action = None
        
        value = math.inf
        for k, v in self.Q[state[0]][state[1]].items():
            if (v < value) and (k in actionSet):
                value = v
                greedy_action = k
        
            
        if greedy_action == None:
            raise Exception('No greedy is generated, check if actionSet is empty')      

        
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
        

       

        #Update the current state-action value
        if reward != None and self.prev_state != None and self.prev_action != None:
            i_prev = self.prev_state[0]
            j_prev = self.prev_state[1]
            a_prev = self.prev_action  
            
            i = state[0]
            j = state[1]
    
            self.Q[i_prev][j_prev][a_prev] = (1-self.alpha)*self.Q[i_prev][j_prev][a_prev] + self.alpha*(reward + self.gamma*self.Q[i][j][greedy_action])

        

        #Set the previous state and action for the next iteration:
        self.prev_state = state
        self.prev_action = action        

   
        return action



    
