#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 07:46:05 2018

@author: diwakar
"""

import numpy as np
import pandas as pd

data= pd.read_csv('set.csv')

X=data.iloc[:,3:7].values
y=data.iloc[:,7].values
tot_test=data.iloc[238425:,2:3].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X, y)


import matplotlib.pyplot as plt
plt.scatter(data.iloc[238425:238945,2:3].values,y_test[0:520],color='red')
plt.plot(data.iloc[238425:238945,2:3].values,y_pred[0:520])
plt.title('Comparision between the Test and Predictedd Values')
plt.xlabel('Total Percentage used')
plt.ylabel('Cores Used')
plt.show()





from sklearn import linear_model
reg = linear_model.Lasso(alpha = 0.1)
reg.fit(X_train,y_train)



lasso_y_pred = reg.predict(X_test)

import matplotlib.pyplot as plt
plt.scatter(data.iloc[238425:238445,2:3].values,y_test[0:20],color='red')
plt.plot(data.iloc[238425:238445,2:3].values,lasso_y_pred[0:20])
plt.title('Comparision between the Test and Predictedd Values')
plt.xlabel('Total Percentage used')
plt.ylabel('Cores Used')
plt.show()










#Linear Regression Check
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Split the data into training/testing sets
diabetes_X_train = X_train
diabetes_X_test = X_test

# Split the targets into training/testing sets
diabetes_y_train = y_train
diabetes_y_test = y_test

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())
plt.show()


#Elastic Nets
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
X, y = make_regression(n_features=2, random_state=0)
regr = ElasticNet(random_state=0)
regr.fit(X_train, y_train)
print(regr.coef_) 
print(regr.intercept_) 
y_pred_elas = regr.predict(X_test)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, y_pred_elas))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, y_pred_elas))

#Ridge Regression
from sklearn import linear_model
reg = linear_model.Ridge (alpha = .5)
reg.fit (X_train,y_train) 

reg.coef_
reg.intercept_ 

y_pred_ridge =reg.predict(X_test)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, y_pred_elas))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, y_pred_elas))




#Graph Plotting

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# X is the 10x10 Hilbert matrix
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)

# #############################################################################
# Compute paths

n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

# #############################################################################
# Display results

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

