
# Find analytic solutions to the Brachistochrone problem.


import numpy as np
import matplotlib.pyplot as plt
import importlib
from math import *


import Agents.Analytic as Analytic


pTs = [(10,9), (10,8), (10,7), (10,6), (10,5), (10,4), (10,3), (10,2), (10,1)]
for pT in pTs:
    x,y = Analytic.agent(pT)    
    plt.plot(x, y)
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.grid()
    
plt.show()




