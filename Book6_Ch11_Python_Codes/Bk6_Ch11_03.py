

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

#%% Heatmap of covariance matrix

SIGMA = y_X_df.cov()

fig, axs = plt.subplots()

h = sns.heatmap(SIGMA,cmap='RdBu_r', linewidths=.05)
h.set_aspect("equal")

vols = np.sqrt(np.diag(SIGMA))

fig, ax = plt.subplots()

plt.bar(labels,vols)
plt.xticks(rotation = 45) 
# Rotates X-Axis Ticks by 45-degrees
plt.ylabel('Daily volatility (standard deviation)')

#%% Heatmap of correlation matrix

fig, ax = plt.subplots()
# Compute the correlation matrix
RHO = y_X_df.corr()

h = sns.heatmap(RHO, cmap="RdBu_r",
            square=True, linewidths=.05,
            annot=False)
h.set_aspect("equal")

fig, ax = plt.subplots()

plt.bar(labels,RHO['SP500'].iloc[:].values)
plt.xticks(rotation = 45) 
# Rotates X-Axis Ticks by 45-degrees
plt.ylabel('Correlation with S&P 500')
RHO.to_excel('corr.xlsx')

#%% Volatility vector space

Angles = np.arccos(RHO)*180/np.pi
fig, axs = plt.subplots()

h = sns.heatmap(Angles, annot=False,cmap='RdBu_r',
                vmin = 30, vmax = 115)
h.set_aspect("equal")
Angles.to_excel('output.xlsx')
#%% Regression

import statsmodels.api as sm

# add a column of ones
X_df = sm.add_constant(X_df)

model = sm.OLS(y_df, X_df)
results = model.fit()
print(results.summary())

p = model.fit().params
print(p)

