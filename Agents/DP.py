import numpy as np
from math import *


class Agent:

    #Note that the following setting can only be changed at creation time (how to enforce this in Python?)
    #"circle" is an obstacle in the path
    def __init__(self, delta_x = 0.5, delta_y = 0.05, range_x = (0,10), range_y = (0,10), circle = None):
        self.delta_x = delta_x
        self.delta_y = delta_y
        
        self.range_x = range_x
        self.range_y = range_y
        
        self.shape = ( round((self.range_y[1]-self.range_y[0])/self.delta_y) + 1, round((self.range_x[1]-self.range_x[0])/self.delta_x) + 1 )
        
        #optimum state value function expressed as a matrix
        self.S = np.zeros(self.shape)
        
        #Each state has a list of possible actions: will this structure be much more efficient in C/C++?
        self.set_A = [[[] for i in range(self.shape[1])] for i in range(self.shape[0])]
        #Remember that j is the time index and i is the distance index
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if not self.checkijInCircle((i,j), circle):
                    for next_y_index in range(self.shape[0]):
                        self.set_A[i][j].append(next_y_index - i)     
    


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



    def print_config(self):
        print("delta_x = ", self.delta_x)
        print("delta_y = ", self.delta_y)
        print("range_x = ", self.range_x)
        print("range_y = ", self.range_y)
        print("S.shape = ", self.S.shape)
        

    def next_state(self, current_state_index, a):
        jj = current_state_index[1] + 1
        ii = current_state_index[0] + a
        return (ii, jj)

    def reward(self, current_state_index, a):
        #return self.F(current_state_index[1]*self.delta_x, current_state_index[0]*self.delta_y+self.delta_y/2, a*self.delta_y/self.delta_x)*self.delta_x
        return self.F2(current_state_index[0]*self.delta_y, a)

    def F(self, x, y, ydesh):
        g = 9.8
        if y==0: return inf
        return sqrt( (1+ydesh**2)/(2*g*y) )

    def F2(self, y, a):
        g = 9.8
        y1 = y
        y2 = y+a*self.delta_y
        accer = g*a*self.delta_y/sqrt((a*self.delta_y)**2 + self.delta_x**2)
        if accer != 0:
            return (sqrt(2*g*y2) - sqrt(2*g*y1))/accer
        elif y1 != 0:
            return self.delta_x/sqrt(2*g*y1)
        else:
            return inf


    def run(self, pT):
        #get a local reference:
        S = self.S
        
        
        #Initialize:
        j = S.shape[1]-1
        for i in range(S.shape[0]):
            S[i,j]=inf
        S[round(pT[1]/self.delta_y),j]=0 #there is a single final state

        #Find optimum value function: S
        for j in range(S.shape[1]-2, -1, -1):
            for i in range(S.shape[0]):
                if S[i, j] == inf: continue                
                minV = inf           
                for a in self.set_A[i][j]:
                    v = self.reward((i,j),a) + S[self.next_state((i,j),a)]
                    if v < minV:
                        minV = v
                S[i,j] = minV

                
        #Find optimum policy with respect to S
        actions = np.zeros(S.shape[1])        
        y = np.zeros(S.shape[1])
        y[0] = 0
        current_state = (0, 0)
        for j in range(S.shape[1]-1):
            i = current_state[0]
            
            minV = inf
            best_action = 0
            for a in self.set_A[i][j]:
                v = self.reward((i,j),a) + S[self.next_state((i,j),a)]
                if v < minV:
                    minV = v
                    best_action = a
            
            actions[j] = best_action
            current_state = self.next_state((i,j),best_action)
            y[j+1] = current_state[0]*self.delta_y
            
            
        x = np.linspace(0, self.delta_x*(S.shape[1]-1), S.shape[1])        

        return (x,y,actions)




