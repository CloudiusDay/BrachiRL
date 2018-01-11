import numpy as np
from math import *
import random
import matplotlib.pyplot as plt
import math




class Agent:

    def __init__(self, delta_x = 0.5, delta_y = 0.05, range_x = (0,10), range_y = (0,10)):
        #The state space is discretized, the state is represented by (i,j), where i is the y axis and j is the x axis. 
        #action is represented by a, the action space at each state (i, j) is defined in self.set_A
        
        #Configure the Marte Carlo Agent:
        self.epsilon = 0.1

        #In an actual application, the actor may not know the state space. Here, it's defined so that we can represent the action value function in a more compact and efficient way.              
        self.delta_x = delta_x
        self.delta_y = delta_y
        
        self.range_x = range_x
        self.range_y = range_y
        
        self.shape = ( round((self.range_y[1]-self.range_y[0])/self.delta_y) + 1, round((self.range_x[1]-self.range_x[0])/self.delta_x) + 1 )
        print('Shape of the state space: ', self.shape)
        
        #Initialize the action value function, which is the average of all returns seen so far, this is what is learnt and should not be intialized between episodes
        self.Q = [[{} for i in range(self.shape[1])] for i in range(self.shape[0])]        
        

        #You will also have to keep the number of state-action pairs that you have seen so far.
        #This should have the exact same number of keys as self.Q       
        self.N = [[{} for i in range(self.shape[1])] for i in range(self.shape[0])] 

        self.number_of_large_greedy_action = 0
        self.new_episode_reset()

        


    def new_episode_reset(self):
        #Initialize the list of returns for the current episode. This is used in the update step.
        #It is being initilialized between episodes.
        #because each state can only be visited once in an episode.             
        self.visited_StateAction = [() for j in range(self.shape[1])]
        self.R = np.zeros(self.shape[1])

        #I also have to store the previous action, initialize this between episodes
        self.prev_action = None
        #self.prev_state = None
        self.ii = 0

        print("number: ", self.number_of_large_greedy_action)   #I can see that over time the number of large greedy actions is getting less and less!
        self.number_of_large_greedy_action = 0

    
    def act(self, state, reward = None, done = False, actionSet = []):    
        #Input state and reward, update self.R and return an action    
        

        #Find the greedy action with respect to the current state action value function
        greedy_action = None
        #Find action from self.Q
        value = math.inf
        for k, v in self.Q[state[0]][state[1]].items():
            if v < value:
                value = v
                greedy_action = k
        
        '''
        if greedy_action != None:
            if greedy_action > 5: greedy_action = 5
            elif greedy_action < -5: greedy_action = -5       
            elif self.prev_action != None:  
                if greedy_action - self.prev_action > 2: greedy_action = self.prev_action + 2
                elif greedy_action - self.prev_action < -2: greedy_action = self.prev_action - 2
        '''
        
        if greedy_action != None and abs(greedy_action) > 5:
            #print("greedy action is too big.")
            self.number_of_large_greedy_action += 1
    
        


        if greedy_action == None and actionSet:
            #print("no close action")
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
            #Instead of picking a random exploring action, pick a random exploring action that is close to the greedy action               
            nearActionSet = []
            for a in actionSet:                
                if abs(a - greedy_action) <= 2:
                    nearActionSet.append(a)
            #print(actionSet)
            #print(nearActionSet)
            if nearActionSet:
                #print("Yes")
                action = np.random.choice(nearActionSet)
            else:
                #print("No")                              
                action = np.random.choice(actionSet)
        
        #if action == None: print("No action is generated!")  #be careful here that you can't use "not action" like you test for empty list 
                  
        
        
        
        #Update the return (accumulated reward):
        self.visited_StateAction[self.ii] = (*state, action)
        if reward: self.R[0:self.ii] += reward
        self.ii += 1     
        

        if done: action = None
   
        self.prev_action = action
        
                
        return action



    def update(self):
        #After each episode, the action value function is updated
        for n in range(self.shape[1]):           
            i,j,a = self.visited_StateAction[n]
            rr = self.R[n] #Note that this is the return, i.e., the accumulated reward, instead of the actual reward
            
            if rr == math.inf: continue

            if a in self.Q[i][j]:
                self.Q[i][j][a] = (self.Q[i][j][a]*self.N[i][j][a] + rr)/(self.N[i][j][a] + 1)
                self.N[i][j][a] += 1
            else:
                self.Q[i][j][a] = rr
                self.N[i][j][a] = 1

        #Adding an explore rate decay:
        #if self.epsilon > 0.001: self.epsilon *= 0.99






