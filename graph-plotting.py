# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 22:59:25 2018

@author: csedi
"""

from matplotlib import pyplot as plt
import numpy as np
import matplotlib
import pandas as pd

data = pd.read_csv('set.csv')
X=data.iloc[:,3:7].values
y=data.iloc[:,7].values

#Core -0
plt.plot(X[1:10000,0:1],y[1:10000],'bo')
plt.xlabel('CPU 0')
plt.ylabel('Total Cores Required')
plt.show()
#Core-1
plt.plot(X[1:10000,1:2],y[1:10000],'bo')
plt.xlabel('CPU 1')
plt.ylabel('Total Cores Required')
plt.show()
#Core-2
plt.plot(X[1:10000,2:3],y[1:10000],'bo')
plt.xlabel('CPU 2')
plt.ylabel('Total Cores Required')
plt.show()
#Core-3
plt.plot(X[1:10000,3],y[1:10000],'bo')
plt.xlabel('CPU 3')
plt.ylabel('Total Cores Required')
plt.show()





#All  in one grapg plots
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2, figsize=(5, 5))
axs[0, 0].plot(X[1:10000,3],y[1:10000],'bo')
axs[1, 0].plot(X[1:10000,2:3],y[1:10000],'bo')
axs[0, 1].plot(X[1:10000,1:2],y[1:10000],'bo')
axs[1, 1].plot(X[1:10000,0:1],y[1:10000],'bo')

plt.show()