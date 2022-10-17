

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns 
from sklearn.datasets import load_iris
from scipy import stats

# Load the iris data
iris_sns = sns.load_dataset("iris") 
# A copy from Seaborn
iris = load_iris()
# A copy from Sklearn

X = iris.data
y = iris.target

feature_names = ['Sepal length, $X_1$','Sepal width, $X_2$',
                 'Petal length, $X_3$','Petal width, $X_4$']

# Convert X array to dataframe
X_df = pd.DataFrame(X, columns=feature_names)

#%% Histograms

# visualize two tails (1%, 99%)

num = 0
fig, axes = plt.subplots(2,2)

for i in [0,1]:
    for j in [0,1]:
        
        sns.histplot(data=X_df, x = feature_names[num], binwidth = 0.2, ax = axes[i][j])
        axes[i][j].set_xlim([0,8]); axes[0][0].set_ylim([0,40])
        
        q1, q50, q99 = np.percentile(X_df[feature_names[num]], [1,50,99])
        axes[i][j].axvline(x=q1, color = 'r')
        axes[i][j].axvline(x=q50, color = 'r')
        axes[i][j].axvline(x=q99, color = 'r')
        
        num = num + 1

# visualize locations of three quartiles

num = 0

fig, axes = plt.subplots(2,2)

for i in [0,1]:
    for j in [0,1]:
        
        sns.histplot(data=X_df, x = feature_names[num], binwidth = 0.2, ax = axes[i][j])
        axes[i][j].set_xlim([0,8]); axes[0][0].set_ylim([0,40])
        
        q75, q50, q25 = np.percentile(X_df[feature_names[num]], [75,50,25])
        axes[i][j].axvline(x=q75, color = 'r')
        axes[i][j].axvline(x=q50, color = 'r')
        axes[i][j].axvline(x=q25, color = 'r')
        
        num = num + 1

#%% KDE +rug plot

num = 0

fig, axes = plt.subplots(2,2)

for i in [0,1]:
    for j in [0,1]:
        
        sns.kdeplot(data=X_df, x = feature_names[num], ax = axes[i][j], fill = True)
        sns.rugplot(data=X_df, x = feature_names[num], ax = axes[i][j], color = 'k', height=.05)
        
        q1, q50, q99 = np.percentile(X_df[feature_names[num]], [1,50,99])
        axes[i][j].axvline(x=q1, color = 'r')
        axes[i][j].axvline(x=q50, color = 'r')
        axes[i][j].axvline(x=q99, color = 'r')
        
        num = num + 1

#%% scatter plot

for i in [1,2,3]:
    
    fig, axes = plt.subplots()
    
    sns.scatterplot(data=X_df, x=feature_names[0], y=feature_names[i])
    sns.rugplot    (data=X_df, x=feature_names[0], y=feature_names[i], height=.05)

#%% pairplot

g = sns.pairplot(X_df)
g.map_upper(sns.scatterplot, color = 'b')
g.map_lower(sns.kdeplot, levels=8, fill=True, cmap="Blues_d") 
g.map_diag(sns.distplot, kde=False, color = 'b')

#%% QQ plot

import pylab

num = 0;

for i in [0,1]:
    for j in [0,1]:
        
        fig, axes = plt.subplots(1,2)
        
        sns.histplot(data=X_df, x = feature_names[num], binwidth = 0.2, ax = axes[0])
        axes[0].set_xlim([0,8]); axes[0].set_ylim([0,40])
        
        values = X_df[feature_names[num]]
        
        stats.probplot(values, dist="norm", plot=pylab)

        plt.xlabel('Normal distribution')
        plt.ylabel('Empirical distribution')
        plt.title(feature_names[num])
        num = num + 1

#%% box plot of data

fig, ax = plt.subplots()
sns.boxplot(data=X_df, palette="Set3", orient="h")

print(X_df.describe())

X_df.quantile(q=[0.25, 0.5, 0.75], axis=0, 
              numeric_only=True, interpolation='midpoint')

#%% combine boxplot and swarmplot

fig, ax = plt.subplots()

sns.boxplot(data=X_df, orient="h", palette="Set3")

sns.swarmplot(data=X_df, 
               linewidth=0.25, orient="h", color=".5")

#%% z score

from scipy import stats

df_zscore = (X_df - X_df.mean())/X_df.std()

# z_score = stats.zscore(X_df)

num = 0

fig, axes = plt.subplots(2,2)

for i in [0,1]:
    for j in [0,1]:
        
        sns.histplot(data=df_zscore, x = feature_names[num], binwidth = 0.2, ax = axes[i][j])
        axes[i][j].set_xlim([-4,4]); axes[i][j].set_ylim([0,40])
        
        axes[i][j].axvline(x=3, color = 'r')
        axes[i][j].axvline(x=2, color = 'r')
        axes[i][j].axvline(x=-3, color = 'r')
        axes[i][j].axvline(x=-2, color = 'r')
        axes[i][j].axvline(x=0, color = 'r')
        
        num = num + 1

#%% Mahal distance

from sklearn.covariance import EllipticEnvelope

clf = EllipticEnvelope(contamination=0.05)

xx, yy = np.meshgrid(np.linspace(3, 9, 50), np.linspace(1, 5, 50))

clf.fit(X_df.values[:,:2])
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, axes = plt.subplots()

plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='r')

plt.scatter(X_df.values[:, 0], X_df.values[:, 1], color='b')

plt.xlim((xx.min(), xx.max()))
plt.ylim((yy.min(), yy.max()))

plt.ylabel(feature_names[0]);
plt.xlabel(feature_names[1]);
plt.gca().set_aspect('equal', adjustable='box')


#%%
