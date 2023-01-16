

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# initializations
import pandas as pd
import pandas_datareader as web
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

tickers = ['^GSPC','TSLA','WMT','MCD','USB',
           'YUM','NFLX','JPM','PFE',
           'F','GM','COST','JNJ'];

stock_levels_df = web.get_data_yahoo(tickers, start = '2020-07-01', end = '2020-12-31')

stock_levels_df = stock_levels_df.rename(columns={"^GSPC": "SP500"})
stock_levels_df.round(2).head()

y_X_df = stock_levels_df['Adj Close'].pct_change()
y_X_df = y_X_df.dropna()


X_df = y_X_df[tickers[1:]]
y_df = y_X_df["SP500"]

labels = ['SP500','TSLA','WMT','MCD','USB',
           'YUM','NFLX','JPM','PFE',
           'F','GM','COST','JNJ'];

#%% Lineplot of stock prices

plt.close('all')

# normalize the initial stock price levels to 1
normalized_stock_levels = stock_levels_df['Adj Close']/stock_levels_df['Adj Close'].iloc[0]

g = sns.relplot(data=normalized_stock_levels,dashes = False,
                kind="line") # , palette="coolwarm"
g.set_xlabels('Date')
g.set_ylabels('Normalized closing price')
g.set_xticklabels(rotation=45)

fig, ax = plt.subplots()
ax = sns.heatmap(y_X_df,
                 cmap='RdBu_r',
                 cbar_kws={"orientation": "vertical"}, 
                 yticklabels=False,
                 vmin = -0.2, vmax = 0.2)
plt.title('[y, X]')

#%% distribution of column features of X

fig, axs = plt.subplots(2,2)

sns.kdeplot(ax = axs[0,0], 
            data=y_X_df[labels[0:4]],
            fill=False, 
            common_norm=False, 
            alpha=.3, linewidth=1,
            palette = "viridis")

axs[0,0].set_xlim([-0.1,0.1])
axs[0,0].set_ylim([0, 45])

sns.kdeplot(ax = axs[0,1], 
            data=y_X_df[labels[4:7]],
            fill=False, 
            common_norm=False, 
            alpha=.3, linewidth=1,
            palette = "viridis")

axs[0,1].set_xlim([-0.1,0.1])
axs[0,1].set_ylim([0, 45])

sns.kdeplot(ax = axs[1,0], 
            data=y_X_df[labels[7:10]],
            fill=False, 
            common_norm=False, 
            alpha=.3, linewidth=1,
            palette = "viridis")

axs[1,0].set_xlim([-0.1,0.1])
axs[1,0].set_ylim([0, 45])

sns.kdeplot(ax = axs[1,1], 
            data=y_X_df[labels[10:]],
            fill=False, 
            common_norm=False, 
            alpha=.3, linewidth=1,
            palette = "viridis")

axs[1,1].set_xlim([-0.1,0.1])
axs[1,1].set_ylim([0, 45])

#%% PCA

from sklearn.decomposition import PCA
pcamodel = PCA(n_components=4)
pca = pcamodel.fit_transform(X_df)

#%% Heatmap of V

V = pcamodel.components_.transpose()

fig, ax = plt.subplots()
ax = sns.heatmap(V,
                 cmap='RdBu_r',
                 xticklabels=['PC1','PC2','PC3','PC4'],
                 yticklabels=list(X_df.columns),
                 cbar_kws={"orientation": "vertical"},
                 vmin=-1, vmax=1,
                 annot = True)
ax.set_aspect("equal")
plt.title('V')

fig, ax = plt.subplots()
ax = sns.heatmap(V.T@V,
                 cmap='RdBu_r',
                 xticklabels=['PC1','PC2','PC3','PC4'],
                 yticklabels=['PC1','PC2','PC3','PC4'],
                 cbar_kws={"orientation": "vertical"},
                 vmin=-1, vmax=1,
                 annot = True)
ax.set_aspect("equal")
plt.title('V.T@V')

# Convert V array to dataframe
V_df = pd.DataFrame(data=V, 
                    columns = ['PC1','PC2','PC3','PC4'], 
                    index   = tickers[1:])

fig, ax = plt.subplots()
sns.lineplot(data=V_df,markers=True, dashes=False,palette = "husl")
plt.axhline(y=0, color='r', linestyle='-')

V_df.to_excel('V.xlsx')


#%%

#%% projected data, Z

Z_df = X_df@V

Z_df = Z_df.rename(columns={0: "PC1", 1: "PC2", 2: "PC3", 3: "PC4"})

fig, ax = plt.subplots()
ax = sns.heatmap(Z_df,
                 cmap='RdBu_r',
                 cbar_kws={"orientation": "vertical"}, 
                 yticklabels=False,
                 vmin = -0.2, vmax = 0.2)
plt.title('Z')

# distribution of column features of Z

fig, ax = plt.subplots()
sns.kdeplot(data=Z_df,fill=False, 
            common_norm=False, 
            alpha=.3, linewidth=1,
            palette = "viridis")
plt.title('Distribution of Z columns')

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

#%% Heatmap of covariance matrix

SIGMA_Z = Z_df.cov()

fig, axs = plt.subplots()

h = sns.heatmap(SIGMA_Z,cmap='RdBu_r', linewidths=.05, annot = True)
h.set_aspect("equal")

#%% approximate X

X_apx = Z_df@V.T

fig, ax = plt.subplots()
ax = sns.heatmap(X_apx,
                 cmap='RdBu_r',
                 cbar_kws={"orientation": "vertical"}, 
                 yticklabels=False,
                 xticklabels = labels[1:],
                 vmin = -0.2, vmax = 0.2)
plt.title('X_apx')


fig, ax = plt.subplots()
ax = sns.heatmap(X_df.to_numpy() - X_apx,
                 cmap='RdBu_r',
                 cbar_kws={"orientation": "vertical"}, 
                 yticklabels=False,
                 xticklabels = labels[1:],
                 vmin = -0.2, vmax = 0.2)
plt.title('Error')


#%%

#%% Least square regression

import statsmodels.api as sm

# add a column of ones
Z_plus_1_df = sm.add_constant(Z_df)

model = sm.OLS(y_df, Z_plus_1_df)
results = model.fit()
print(results.summary())

p_Z = model.fit().params
print(p_Z)

#%% coefficients

b_Z = p_Z[1:].T
b_X = V@b_Z

b_X_df = pd.DataFrame(data=b_X.T, index = tickers[1:])

fig, ax = plt.subplots()
b_X_df.plot.bar()

b0 = y_df.mean() - X_df.mean().T@b_X


#%%

#%% increasing number of principal components

coeff_df = pd.DataFrame()
explained_array = []

num_PCs = [4,5,6,7,8,9]

for num_PC in num_PCs:
    
    pcamodel = PCA(n_components=num_PC)
    pca = pcamodel.fit_transform(X_df)
    V = pcamodel.components_.transpose()
    Z_df = X_df@V

    Z_plus_1_df = sm.add_constant(Z_df)
    model = sm.OLS(y_df, Z_plus_1_df)
    p_Z = model.fit().params
    
    b_Z = p_Z[1:].T
    b_X = V@b_Z
    b_X_df = pd.DataFrame(data=b_X.T, index = tickers[1:], columns = ['PC1~' + str(num_PC)])
    explained = np.sum(pcamodel.explained_variance_ratio_)
    print(explained)
    
    explained_array.append(explained)
    
    coeff_df = pd.concat([coeff_df, b_X_df], axis = 1)


fig, ax = plt.subplots()
plt.bar(num_PCs, explained_array)

fig, ax = plt.subplots()
h = sns.lineplot(data=coeff_df,markers=True, dashes=False,palette = "husl")
plt.axhline(y=0, color='r', linestyle='-')
h.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

fig, ax = plt.subplots()
h = sns.lineplot(data=coeff_df.T,markers=True, dashes=False,palette = "husl")
plt.axhline(y=0, color='r', linestyle='-')
h.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

