import numpy as np
from math import *
import random
import matplotlib.pyplot as plt
import math


#Alternatively, I can use a virtual 't' that is discrete, this virtual 't' does not correspond to time, 
#In this case, both x and y are truly state variables and both can be made continuous by using FA!



class Env:
    
    def __init__(self, y_final, delta_x = 0.5, range_x = (0,10), circle = None):
        #The time step is still discretized but the state variable y and action is continuous. State is represented as (x,y)        
        #The allowed action at state s = (x,y) is defined by an inequality (check this in a function)
        #x is incremented in discrete step while y is changed in continuously.        

        
        #If y_final > 0 then there is no solution. Not sure if you want to check that here
        #Check that range_x is divisible by delta_x           
                
        self.circle = circle #input additional constraint

        self.delta_x = delta_x        
        self.range_x = range_x

        self.final_state = (self.range_x[1], y_final)        

        self.reset()        

    

          
    #Have to call this first before stepping:
    def reset(self):
        self.state = (0,0)
        self.accumulated_reward = 0        




    def checkOutOfCircle(self, s):
        if self.circle == None:
            return True
        else:
            x = s[0]
            y = s[1]
            
            x0 = circle[0]
            y0 = circle[1]
            r = circle[2]
            
            return (x - x0)**2 + (y - y0)**2 > r**2




    def checkValidState(self, s):
        x = s[0]
        y = s[1]

        #check x within range
        flag1 = (x >= self.range_x[0] and x <= self.range_x[1])       
        #check y within range
        flag2 = (y >= 0)
        #check additional contraint
        flag3 = self.checkOutOfCircle(s)
        return (flag1 and flag2 and flag3)

  


  
    def checkValidAction(self, a):
        #This is the valid action for the current state: self.state
        next_state = (self.state[0] + self.delta_x, self.state[1] + action)
        return self.checkValidState(next_state)
    



    
    def step(self, action):
        if self.state[0] == self.final_state[0]:
            return self.state, 0, True  #if you are already at the final state then you can't advance further.
        if action == None:
            raise Exception('Cant step the env because action is None.')
        
        next_state = self.advanceState(action)
        if next_state == None:
            raise Exception('next state is None, the action is probably invalid.')

        reward = self.F(self.state[0], action)
        
        self.state = next_state
        self.accumulated_reward += reward
        if self.state[0] == self.final_state[0]:
            done = True
        else:
            done = False
        return next_state, reward, done





    def advanceState(self, action):
        next_state = (self.state[0] + self.delta_x, self.state[1] + action)
        #print(self.state, action, next_state)        
        if self.checkValidState(next_state):
            return next_state
        else:    
            return None



    def render(self):
        pass        




    def F(self, y, a):
        g = 9.8
        y1 = y
        y2 = y+a
        if y2<0: y2=0
        accer = g*a/sqrt(a**2 + self.delta_x**2)
        if accer != 0:
            return (sqrt(2*g*y2) - sqrt(2*g*y1))/accer
        elif y1 != 0:
            return self.delta_x/sqrt(2*g*y1)
        else:
            return math.inf



    
    def sample_action(self): 
        pass
    







