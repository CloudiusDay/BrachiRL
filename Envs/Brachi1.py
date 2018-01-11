import numpy as np
from math import *
import random
import matplotlib.pyplot as plt
import math



class Env:
    
    def __init__(self, y_final, delta_x = 0.5, delta_y = 0.05, range_x = (0,10), range_y = (0,10), circle = None):
        #The state space is discretized, the state is represented by (i,j), where i is the y axis and j is the x axis. 
        #action is represented by a, the action space at each state (i, j) is defined in self.set_A
        
        assert y_final > range_y[0] and y_final < range_y[1]            

        self.delta_x = delta_x
        self.delta_y = delta_y
        
        self.range_x = range_x
        self.range_y = range_y
        
        self.shape = ( round((self.range_y[1]-self.range_y[0])/self.delta_y) + 1, round((self.range_x[1]-self.range_x[0])/self.delta_x) + 1 )
        
        self.final_state = (round(y_final/self.delta_y), self.shape[1]-1)        

        #Each state has a list of possible actions: will this structure be much more efficient in C/C++?
        self.set_A = [[[] for i in range(self.shape[1])] for i in range(self.shape[0])]
        for i in range(self.shape[0]):
            for j in range(self.shape[1]-2):            
                if not self.checkijInCircle((i,j), circle):
                    for next_y_index in range(self.shape[0]):
                         self.set_A[i][j].append(next_y_index - i)
              
        
        j = self.shape[1]-1
        #In the final time stamp, the only available state is the final state and there is no more action to take
        #No need to do anything here as it is already initialize to empty list

        j = self.shape[1]-2
        for i in range(self.shape[0]):
            #In the time stamp that leads to the final time stamp, there is only one action available for each state. 
            self.set_A[i][j].append(self.final_state[0] - i)
        
        
        self.reward_accum = 0


    def checkijInCircle(self, ij, circle):
        if circle == None:
            return False
        else:
            x = ij[1]*self.delta_x
            y = ij[0]*self.delta_y
            
            x0 = circle[0]
            y0 = circle[1]
            r = circle[2]
            
            return (x - x0)**2 + (y - y0)**2 < r**2



    
    def step(self, action):
        if self.state == self.final_state:
            return self.state, 0, True  #if you are already at the final state then you can't advance further.
        if action == None:
            action = 0  #No action is not an option, the time stamp always has to increase!
        
        next_state = (self.state[0]+action, self.state[1]+1)
        reward = self.F(self.state[0]*self.delta_y, action)
        self.state = next_state
        self.accumulated_reward += reward
        if self.state == self.final_state:
            done = True
        else:
            done = False
        return next_state, reward, done




    #have to call this first before stepping:
    def reset(self):
        self.state = (0,0)
        self.accumulated_reward = 0    



    def render(self):
        pass        






    def F(self, y, a):
        g = 9.8
        y1 = y
        y2 = y+a*self.delta_y
        accer = g*a*self.delta_y/sqrt((a*self.delta_y)**2 + self.delta_x**2)
        if accer != 0:
            return (sqrt(2*g*y2) - sqrt(2*g*y1))/accer
        elif y1 != 0:
            return self.delta_x/sqrt(2*g*y1)
        else:
            return math.inf

    
    def sample_action(self): 
        if self.set_A[self.state[0]][self.state[1]]:        
            return random.choice(self.set_A[self.state[0]][self.state[1]])    
        else:
            return None
    









