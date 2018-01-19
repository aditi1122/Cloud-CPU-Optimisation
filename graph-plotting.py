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


x=X[0:200,0:1]
y=y[0:200]



plt.scatter(x, y, marker='o', s=5, zorder=10)

plt.title('griddata test (%d points)' % npts)
plt.show()