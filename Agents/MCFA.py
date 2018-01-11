import numpy as np
from math import *
import random
import matplotlib.pyplot as plt
import math

#This initial version on Function Approximation (FA) does not handle additional constraint yet


class Agent:

    def __init__(self, y_final, range_y = (0,10), delta_x = 0.5, range_x = (0,10)):
        #The state is represented as (x, y), x is discretized (as this is the time stamp) 
        #but y is continuous and so is the action. For simplicity, I'm going to treat all three as 
        #continuous variables.
        
        #Configure the Monte Carlo Agent:
        self.epsilon = 0.1 #For the epsilon greedy policy
        self.alpha = 0.00001  #learning rate


        #Make sure the time step is discretized in the same way:
        self.delta_x = delta_x   
        self.range_x = range_x
        self.max_num_steps = round((self.range_x[1]-self.range_x[0])/self.delta_x) + 1
         
        self.y_final = y_final       
        self.range_y = range_y


        #Initialize the weight for the function approximation
        self.theta=np.random.rand(15)
        #Is there a way I can initialize this to guarantee that the stationary point is a minimum point?
        #Is it necessary to guarantee this?


        self.new_episode_reset()


    def new_episode_reset(self):
        self.visited_StateAction = [() for j in range(self.max_num_steps)]
        self.R = np.zeros(self.max_num_steps)
        self.ii = 0

        self.prev_action = None
        self.prev_state = None
    

    def act(self, state, done, reward = None):
        
        if state == None: raise Exception('There is no observation to make a decision.')
        if done: return None


        #Find the greedy action with respect to the current state action value function
        greedy_action = None
        a = []
        a.append(self.range_y[0] - state[1])
        a.append(self.range_y[1] - state[1])        
        sa = self.stationary_qhat(state)
        if state[1] + sa >= self.range_y[0] and state[1] + sa <= self.range_y[1]:
            a.append(sa)
        
        value = math.inf        
        for aa in a:
            v = self.qhat(state, aa)
            if v < value:
                value = v
                greedy_action = aa
        
        if greedy_action != sa:
            greedy_action = random.uniform(self.range_y[0] - state[1], self.range_y[1] - state[1])        


        if greedy_action == None:
            raise Exception('No greedy action is generated.')      
    



        #Find the self.epsilon soft greedy action:
        action = None         
        if abs(state[0] + self.delta_x - self.range_x[1]) < self.delta_x/4:
            action = self.y_final - state[1]
        else:
            prob = 1-self.epsilon+self.epsilon/10  #a bit of a magic number here
            if np.random.uniform() <= prob:
                action = greedy_action  
            else:            
                action = random.uniform(self.range_y[0] - state[1], self.range_y[1] - state[1])
        
        
        if action == None:
            raise Exception('No action is generated. Check range_y.')      



        #Update the return (accumulated reward):
        if self.prev_action != None and self.prev_state != None:
            self.visited_StateAction[self.ii] = (*self.prev_state, self.prev_action)
            if reward: self.R[0:self.ii] += reward
            self.ii += 1 
        
        

        self.prev_action = action
        self.prev_state = state


        return action




    #After each episode, the action value function is updated
    def update(self):        
        for n in range(self.ii):           
            x,y,a = self.visited_StateAction[n]
            s = (x,y)
            rr = self.R[n] #Note that this is the return, i.e., the accumulated reward, instead of the actual reward            
            if rr != math.inf:
                self.theta += self.alpha*(rr - self.qhat(s,a))*self.dqhat_dtheta(s,a)            



    def qhat_terms(self, s, a):
        vec = np.zeros(15)
        x,y = s
        vec[0] = x
        vec[1] = y
        vec[2] = a
        
        vec[3] = x**2
        vec[4] = y**2
        vec[5] = a**2
        vec[6] = x*y
        vec[7] = x*a
        vec[8] = y*a
        
        vec[9] = x**2*y
        vec[10] = x**2*a    
        vec[11] = y**2*x
        vec[12] = y**2*a
        vec[13] = a**2*x
        vec[14] = a**2*y

        return vec

    #Calculate the state-action value based on self.theta
    def qhat(self, s, a):
        return np.sum(self.qhat_terms(s,a)*self.theta)    



    #Calculate the gradient of qhat respect to theta
    def dqhat_dtheta(self, s, a):
        return self.qhat_terms(s,a)



    #Calculate the stationary point of qhat
    def stationary_qhat(self, s):
        x,y = s
        A = self.theta[5] + self.theta[13]*x + self.theta[14]*y
        B = self.theta[2] + self.theta[7]*x + self.theta[8]*y + self.theta[10]*x**2 + self.theta[12]*y**2
        return -B/(2*A)



 


