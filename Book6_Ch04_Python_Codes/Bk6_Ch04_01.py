

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

# Load the iris data
iris_sns = sns.load_dataset("iris") 
# A copy from Seaborn
iris = load_iris()
# A copy from Sklearn

X = iris.data
y = iris.target

feature_names = ['Sepal length, x1','Sepal width, x2',
                 'Petal length, x3','Petal width, x4']

# Convert X array to dataframe
X_df = pd.DataFrame(X, columns=feature_names)

#%% visualize original data

# Heatmap of X

plt.close('all')
sns.set_style("ticks")

X = X_df.to_numpy();

# Visualize the heatmap of X

fig, ax = plt.subplots()
ax = sns.heatmap(X,
                 cmap='RdYlBu_r',
                 xticklabels=list(X_df.columns),
                 cbar_kws={"orientation": "vertical"},
                 vmin=-1, vmax=9)
plt.title('X')

# distribution of column features of X

fig, ax = plt.subplots()
sns.kdeplot(data=X,fill=True, 
            common_norm=False, 
            alpha=.3, linewidth=1,
            palette = "viridis")
plt.title('Distribution of X columns')

# violin plot of data

fig, ax = plt.subplots()

sns.violinplot(data=X_df, palette="Set3", bw=.2,
               cut=1, linewidth=0.25, inner="points", orient="v")
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

# parallel coordinates

fig, ax = plt.subplots()
# Make the plot
pd.plotting.parallel_coordinates(iris_sns, 'species', colormap=plt.get_cmap("Set2"))
plt.show()

#%% Demean, centralize 

X_demean = X_df.sub(X_df.mean())

# distribution of column features of X

fig, ax = plt.subplots()
ax = sns.heatmap(X_demean,
                 cmap='RdYlBu_r',
                 xticklabels=list(X_df.columns),
                 cbar_kws={"orientation": "vertical"},
                 vmin=-3, vmax=3)
plt.title('$X_{demean}$')

fig, ax = plt.subplots()
sns.kdeplot(data=X_demean,fill=True, 
            common_norm=False, 
            alpha=.3, linewidth=1,
            palette = "viridis")
plt.title('Distribution of $X_{demean}$ columns')

# violin plot of centralized data

fig, ax = plt.subplots()

sns.violinplot(data=X_demean, palette="Set3", bw=.2,
               cut=1, linewidth=0.25, inner="points", orient="v")
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

# parallel coordinates 

iris_df_demean = X_demean.copy()
iris_df_demean['species'] = iris_sns['species']

fig, ax = plt.subplots()

pd.plotting.parallel_coordinates(iris_df_demean, 
                                 'species', 
                                 colormap=plt.get_cmap("Set2"))
plt.show()

#%% Standardize

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(X_df)
# Z_score = scaler.transform(X_df)

Z_score = (X_df - X_df.mean()) /X_df.std()

fig, ax = plt.subplots()
ax = sns.heatmap(Z_score,
                 cmap='RdYlBu_r',
                 xticklabels=list(X_df.columns),
                 cbar_kws={"orientation": "vertical"},
                 vmin=-3, vmax=3)
plt.title('Z')


# KDE plot of normalized data

fig, ax = plt.subplots()
sns.kdeplot(data=Z_score,fill=True, 
            common_norm=False, 
            alpha=.3, linewidth=1,
            palette = "viridis")
plt.title('Distribution of $X_{demean}$ columns')

# violin plot of normalized data

fig, ax = plt.subplots()

sns.violinplot(data=Z_score, palette="Set3", bw=.2,
               cut=1, linewidth=0.25, inner="points", orient="v")
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

# parallel coordinates 

iris_df_z_scores = Z_score.copy()
iris_df_z_scores['species'] = iris_sns['species']

fig, ax = plt.subplots()

pd.plotting.parallel_coordinates(iris_df_z_scores, 
                                 'species', 
                                 colormap=plt.get_cmap("Set2"))
plt.show()

#%% normalize
# similar function: sklearn.preprocessing.minmax_scale
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(X_df)
# X_normalized = scaler.transform(X_df)

X_normalized = (X_df - X_df.min()) /(X_df.max() - X_df.min())

fig, ax = plt.subplots()
ax = sns.heatmap(X_normalized,
                 cmap='RdYlBu_r',
                 xticklabels=list(X_df.columns),
                 cbar_kws={"orientation": "vertical"},
                 vmin=0, vmax=1)
plt.title('Normalized')


# KDE plot of normalized data

fig, ax = plt.subplots()
sns.kdeplot(data=X_normalized,fill=False, 
            common_norm=False, 
            alpha=.3, linewidth=1,
            palette = "viridis")
plt.title('Distribution of $X_{demean}$ columns')

# violin plot of normalized data

fig, ax = plt.subplots()

sns.violinplot(data=X_normalized, palette="Set3", bw=.2,
               cut=1, linewidth=0.25, inner="points", orient="v")
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

# parallel coordinates 

iris_df_normalized = X_normalized.copy()
iris_df_normalized['species'] = iris_sns['species']

fig, ax = plt.subplots()
pd.plotting.parallel_coordinates(iris_df_normalized, 
                                 'species', 
                                 colormap=plt.get_cmap("Set2"))
plt.show()

