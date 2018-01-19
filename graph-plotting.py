# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 22:59:25 2018

@author: csedi
"""

from matplotlib import pyplot as plt
import numpy as np
import matplotlib
import pandas as pd

data = pd.read_csv('set2.csv')
X=data.iloc[:,3:7].values
y=data.iloc[:,7].values

#Core -0

x=X[0:20,0:1]
y=y[0:20]


plt.xlabel("Core 0")
plt.ylabel("Actual Requiress")
plt.scatter(x, y, marker='o',edgecolors='green')
plt.title('Core 0 - Actual')




x1=X[0:20,0:1]

plt.xlabel("Core 0")
plt.ylabel("Actual Requiress")
plt.scatter(x1, y, marker='o',edgecolors='red')

plt.title('Core 0 - Actual')
plt.show()


plt.figure(1)
plt.subplot()
plt.plot(x,y,marker='o')

plt.figure(2)
plt.subplot()
plt.plot(x1, y, marker='o')

plt.show()