#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 18:46:48 2018

@author: diwakar
"""

import pandas as pd
data=pd.read_csv('Diwakar.csv')
X=data.iloc[:,[0,1]].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Encoding the Days
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
#Encoding the Time
labelencoder_X = LabelEncoder()
X[:, 6] = labelencoder_X.fit_transform(X[:, 6])
onehotencoder = OneHotEncoder(categorical_features = [6])
X = onehotencoder.fit_transform(X).toarray()


#Dependent Variable
y=data.iloc[:,8].values




#Train test Split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Scores Calculator 
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X, y)


import matplotlib.pyplot as plt
plt.scatter(data.iloc[238425:238945,2:3].values,y_test[0:520],color='red')
plt.plot(data.iloc[238425:238945,2:3].values,y_pred[0:520])
plt.title('Comparision between the Test and Predictedd Values')
plt.xlabel('Total Percentage used')
plt.ylabel('Cores Used')
plt.show()



#Linear Regression Check
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
diabetes_X_train = X_train
diabetes_X_test = X_test
diabetes_y_train = y_train
diabetes_y_test = y_test

regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
diabetes_y_pred = regr.predict(diabetes_X_test)
print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())
plt.show()


#Lasso Regression Models
from sklearn import linear_model
reg = linear_model.Lasso(alpha = 0.1)
reg.fit(X_train,y_train)
lasso_y_pred = reg.predict(X_test)
reg.coef_
reg.intercept_
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, lasso_y_pred))
print('Variance score: %.2f' % r2_score(diabetes_y_test, lasso_y_pred))



#Elastic Nets
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
X, y = make_regression(n_features=2, random_state=0)
regr = ElasticNet(random_state=0)
regr.fit(X_train, y_train)
print(regr.coef_) 
print(regr.intercept_) 
y_pred_elas = regr.predict(X_test)

print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, y_pred_elas))
print('Variance score: %.2f' % r2_score(diabetes_y_test, y_pred_elas))

#Ridge Regression
from sklearn import linear_model
reg = linear_model.Ridge (alpha = 5)
reg.fit (X_train,y_train) 
reg.coef_
reg.intercept_ 
y_pred_ridge =reg.predict(X_test)
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, y_pred_ridge))
print('Variance score: %.2f' % r2_score(diabetes_y_test, y_pred_ridge))


#Graph Plotting for Ridge Regression
import matplotlib.pyplot as plt
from sklearn import linear_model

# Compute paths

n_alphas = 5
alphas = np.logspace(-10, -2, n_alphas)
coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()



# SDG Regressor
from sklearn import linear_model
clf = linear_model.SGDRegressor()
clf.fit(X_train, y_train)
clf.coef_
clf.intercept_
y_pred_sdg = clf.predict(X_test)
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, y_pred_sdg))
print('Variance score: %.2f' % r2_score(diabetes_y_test, y_pred_sdg))
