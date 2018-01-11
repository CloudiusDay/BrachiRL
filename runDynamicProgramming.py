


# 1) finds the dynamic programing solution to the Brachistochrone problem 
# 2) compares it with the analytic solution
# 3) shows the advantage of using dynamic programming instead of solving the problem analytically.


# In[2]:

import numpy as np
import matplotlib.pyplot as plt
import importlib
from math import *


import Agents.Analytic as Analytic
import Agents.DP as DP







# In[3]:

# Dynamic Programming (DP) Solution


# In[4]:

agentDP1 = DP.Agent()
agentDP1.print_config()


# In[5]:

p0 = (0,0)
pT = (10, 3)

x,y,_ = agentDP1.run(pT)

plt.figure(1)
plt.plot(x, y)
plt.axis('equal')
plt.gca().invert_yaxis()
plt.grid()



# In[6]:

# Compare DP with the Analytic Solution


# In[7]:

p0 = (0,0)
pT = (10, 3)

x,y = Analytic.agent(pT)
plt.figure(2)    
plt.plot(x, y)
plt.axis('equal')
plt.gca().invert_yaxis()
plt.grid()

agentDP2 = DP.Agent(range_y = (0,10))
x,y,_ = agentDP2.run(pT)    
plt.plot(x, y)



# In[8]:

# The advantage of using DP is that it allows us to add arbitrary constraint 
# without going through pages of derivation again to arrive at a new analytical solution.


# In[9]:

agentDP3 = DP.Agent(circle=(3,3,2))
x,y,_ = agentDP3.run(pT)
plt.figure(3)    
plt.plot(x, y)
plt.axis('equal')
plt.gca().invert_yaxis()
plt.grid()

x,y,_ = agentDP2.run(pT)
plt.plot(x, y)


plt.show()





