

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

feature_names = ['Sepal length','Sepal width',
                 'Petal length','Petal width']

# Convert X array to dataframe
X_df = pd.DataFrame(X, columns=feature_names)

#%% Heatmap of X

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

#%% Pairplot of the original data X

# with no class labels
g = sns.pairplot(iris_sns)
g.map_upper(sns.scatterplot, color = 'b')
g.map_lower(sns.kdeplot, levels=8, fill=True, cmap="Blues_d") 
g.map_diag(sns.distplot, kde=False, color = 'b')

# with class labels
g = sns.pairplot(iris_sns,hue="species", plot_kws={"s": 6}, palette = "viridis")
g.map_lower(sns.kdeplot)

#%%

#%% PCA

from sklearn.decomposition import PCA

pcamodel = PCA(n_components=4)
pca = pcamodel.fit_transform(X_df)

#%% Eigen decomposition

X_SIGMA = np.cov(X.T)
X_RHO   = np.corrcoef(X.T)

Lambda,V_eigen = np.linalg.eig(X_SIGMA)

# V_eigen@np.diag(Lambda)@(V_eigen.T)
# np.savetxt('X_SIGMA.csv', X_SIGMA, delimiter=',')

X_sigma = np.std (X, axis=0)
X_VAR   = np.var (X, axis=0)
X_mu    = np.mean(X, axis=0)

#%% SVD decomposition

U_SVD, S_SVD, V_T_SVD = np.linalg.svd(X,full_matrices = False)


#%% Heatmap of V transpose

fig, ax = plt.subplots()
ax = sns.heatmap(pcamodel.components_,
                 cmap='RdYlBu_r',
                 yticklabels=['PC1','PC2','PC3','PC4'],
                 xticklabels=list(X_df.columns),
                 cbar_kws={"orientation": "vertical"},
                 vmin=-1, vmax=1)
ax.set_aspect("equal")
plt.title('V transpose')

#%% Heatmap of V

V = pcamodel.components_.transpose()

fig, ax = plt.subplots()
ax = sns.heatmap(V,
                 cmap='RdYlBu_r',
                 xticklabels=['PC1','PC2','PC3','PC4'],
                 yticklabels=list(X_df.columns),
                 cbar_kws={"orientation": "vertical"},
                 vmin=-1, vmax=1)
ax.set_aspect("equal")
plt.title('V')

# Convert V array to dataframe
V_df = pd.DataFrame(data=V, 
                    columns = ['PC1','PC2','PC3','PC4'], 
                    index   = ['Sepal length','Sepal width',
                               'Petal length','Petal width'])

fig, ax = plt.subplots()
sns.lineplot(data=V_df,markers=True, dashes=False,palette = "husl")
plt.axhline(y=0, color='r', linestyle='-')

#%% V.T @ V = I

fig, ax = plt.subplots()
ax = sns.heatmap((V.T)@V,
                 cmap='RdYlBu_r',
                 xticklabels=[ "PCA"+str(x) for x in range(1,pcamodel.n_components_+1)],
                 yticklabels=list(X_df.columns),
                 cbar_kws={"orientation": "vertical"},
                 vmin=-1, vmax=1)
ax.set_aspect("equal")
plt.title('V.T @ V = I')


#%%

#%% Heatmap of Z

# Project original data X to Z
Z = X@V


fig, ax = plt.subplots()
ax = sns.heatmap(Z,
                 cmap='RdYlBu_r',
                 xticklabels=['PC1','PC2','PC3','PC4'],
                 cbar_kws={"orientation": "vertical"},
                 vmin=-1, vmax=9)
plt.title('Z')

fig, ax = plt.subplots()
sns.kdeplot(data=Z,fill=True, 
            common_norm=False, 
            alpha=.3, linewidth=1,
            palette = "viridis")
plt.title('Distribution of Z columns')

# Calculate statistics of Z

Z_SIGMA = np.cov(Z.T)
Z_RHO   = np.corrcoef(Z.T)

Z_sigma = np.std (Z, axis=0)
Z_VAR   = np.var (Z, axis=0)
Z_mu    = np.mean(Z, axis=0)

#%% heatmap of covariance and correlation matrices

fig, ax = plt.subplots()
ax = sns.heatmap(Z_SIGMA,
                 cmap='RdBu_r',
                 xticklabels=[ "PCA"+str(x) for x in range(1,pcamodel.n_components_+1)],
                 yticklabels=[ "PCA"+str(x) for x in range(1,pcamodel.n_components_+1)],
                 cbar_kws={"orientation": "vertical"}, 
                 annot = True)

ax.set_aspect("equal")
ax.set_title("Covariance matrix of Z")

fig, ax = plt.subplots()
ax = sns.heatmap(Z_RHO,
                 cmap='RdBu_r',
                 xticklabels=[ "PCA"+str(x) for x in range(1,pcamodel.n_components_+1)],
                 yticklabels=[ "PCA"+str(x) for x in range(1,pcamodel.n_components_+1)],
                 cbar_kws={"orientation": "vertical"},annot = True)

ax.set_aspect("equal")
ax.set_title("Correlation matrix of Z") 


#%% Pairplot of the original data Z


PCA_df = pd.DataFrame(data=pca, columns=["PC1", "PC2","PC3", "PC4"])
PCA_df['species'] = iris_sns['species']

# with no class labels

g = sns.pairplot(PCA_df)
g.map_upper(sns.scatterplot, color = 'g')
g.map_lower(sns.kdeplot, levels=8, fill=True, cmap="Greens_d") 
g.map_diag(sns.distplot, kde=False, color = 'g')

# with class labels

g = sns.pairplot(PCA_df,hue="species", plot_kws={"s": 6},palette = "husl")
g.map_lower(sns.kdeplot)


#%%
#%% Heatmap of X1~X4

X_re = np.zeros_like(X);
# Reproduce original data X

for i in range(4):

    z_i = np.array([Z[:,i]]).T
    v_i = np.array([V[:,i]]).T
    
    X_i = z_i@(v_i.transpose())
    X_re = X_re + X_i;
    fig, ax = plt.subplots()
    ax = sns.heatmap(X_i,
                     cmap='RdYlBu_r',
                     xticklabels=list(X_df.columns),
                     cbar_kws={"orientation": "vertical"},
                     vmin=-1, vmax=9)
    # # ax.set_aspect("equal")
    plt.title('X_' + str(i+1))


#%% X1 + X2 + X3 + X4 to reproduce X

fig, ax = plt.subplots()
ax = sns.heatmap(X_re,
                 cmap='RdYlBu_r',
                 xticklabels=list(X_df.columns),
                 cbar_kws={"orientation": "vertical"},
                 vmin=-1, vmax=9)
plt.title('X reproduced')
# Visualize reproduced X

#%% X1 + X2 to approximate the original data

z_12 = Z[:,0:2]
v_12 = V[:,0:2]

X_12 = z_12@(v_12.transpose())

fig, ax = plt.subplots()
ax = sns.heatmap(X_12,
                 cmap='RdYlBu_r',
                 cbar_kws={"orientation": "vertical"},
                 vmin=-1, vmax=9)
plt.title('X1 + X2')


fig, ax = plt.subplots()
ax = sns.heatmap(X - X_12,
                 cmap='RdYlBu_r',
                 cbar_kws={"orientation": "vertical"}, 
                 vmin=-1, vmax=9)
plt.title('Error, E')


#%% biplot

from yellowbrick.features import PCA
from yellowbrick.style import set_palette
set_palette('pastel')

fig, ax = plt.subplots()
visualizer = PCA(scale=True, proj_features=True)
visualizer.fit_transform(iris_sns[[
    'sepal_length', 'sepal_width',
    'petal_length','petal_width']], y)

visualizer.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
visualizer = PCA(scale=True, proj_features=True,projection = 3)
visualizer.fit_transform(iris_sns[[
    'sepal_length', 'sepal_width',
    'petal_length','petal_width']], y)

visualizer.show()


#%%

#%% Scree plot

# pcamodel.explained_variance_
# pcamodel.explained_variance_ratio_

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Principal component')
ax1.set_ylabel('Variance explained (%)', color=color)
plt.plot(range(1,len(pcamodel.explained_variance_ratio_ )+1),
         np.cumsum(pcamodel.explained_variance_ratio_,),
         color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([0,1])

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Variance', color=color)
plt.bar(range(1,len(pcamodel.explained_variance_ )+1),pcamodel.explained_variance_ )

ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout() 
# plt.show()


#%%

