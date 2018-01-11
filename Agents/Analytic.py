import numpy as np
from math import *
from scipy.optimize import newton


def ff(theta):
    return (theta - np.sin(theta))/(1-np.cos(theta))
    



#Assume the starting point is (0,0)
def agent(pT, N = 500):
    
    gg = lambda theta: ff(theta) - pT[0]/pT[1]

    thetaT = newton(gg, 2*np.pi-0.1)
    k2 = 2*pT[0]/(thetaT - np.sin(thetaT))
    
    theta = np.linspace(0, thetaT, N)
    x = 0.5*k2*(theta - np.sin(theta))
    y = 0.5*k2*(1 - np.cos(theta))
    
    return (x, y)







