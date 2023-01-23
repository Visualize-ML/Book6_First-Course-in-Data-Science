

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X_, y = load_iris(return_X_y=True)
X = X_[:,0]
X = X[:, np.newaxis]

#%% KDE distributions of three classes 

import seaborn as sns

iris = sns.load_dataset("iris")

fig, ax = plt.subplots()

sns.kdeplot(data=iris[['sepal_length','species']], x='sepal_length',
            hue = 'species',
            palette = "viridis",alpha=.3, linewidth=1,
            fill=False)
plt.xlabel('Sepal length, $x_1$')
plt.xticks([4,5,6,7,8])

fig, ax = plt.subplots()

plt.scatter(X.ravel(), y, s = 8, alpha = 0.5, label = 'Original')
plt.ylabel('Real y')
plt.xlabel('Sepal length, $x_1$')
plt.yticks([0,1,2])
plt.xticks([4,5,6,7,8])
plt.legend()
plt.tight_layout()
plt.show()

#%% logistic regression

import numpy as np

clf = LogisticRegression()
clf.fit(X, y)

X_test = np.linspace(X.min()*0.9,X.max()*1.1,num = 100)
X_test = X_test[:, np.newaxis]

y_hat = clf.predict(X_test)

y_prob = clf.predict_proba(X_test)

b1 = clf.coef_
b0 = clf.intercept_


#%% probabilities

x = np.linspace(X.min()*0.9,X.max()*1.1,num = 100);

fig, ax = plt.subplots()
plt.plot(X_test, y_prob[:,0], color='r', 
         linewidth=1, label = 'Class 0')
plt.fill_between(x, y_prob[:,0], color='r', alpha = 0.5)

plt.plot(X_test, y_prob[:,1], color='b', 
         linewidth=1, label = 'Class 1')
plt.fill_between(x, y_prob[:,1], color='b', alpha = 0.5)

plt.plot(X_test, y_prob[:,2], color='g', 
         linewidth=1, label = 'Class 2')
plt.fill_between(x, y_prob[:,2], color='g', alpha = 0.5)

plt.ylabel('Probability')
plt.xlabel('Sepal length, $x_1$')
plt.xticks([4,5,6,7,8])
plt.legend()
plt.tight_layout()
plt.show()

#%% Predicted y

fig, ax = plt.subplots()

plt.scatter(X.ravel(), clf.predict(X), s = 8, 
            alpha = 0.5, label = 'Predicted')

plt.ylabel('Predicted y')
plt.xlabel('Sepal length, $x_1$')
plt.yticks([0,1,2])
plt.xticks([4,5,6,7,8])
plt.legend()
plt.tight_layout()
plt.show()
