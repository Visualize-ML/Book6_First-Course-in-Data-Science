

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


# bi-variate regression

# initializations and download results 
import pandas as pd
import pandas_datareader as web
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas_datareader as web
import statsmodels.api as sm


y_levels_df = web.get_data_yahoo(['^GSPC'], start = '2020-01-01', end = '2020-12-31')

y_levels_df.round(2).head()
y_df = y_levels_df['Adj Close'].pct_change()
y_df = y_df.dropna()


X_levels_df = web.get_data_yahoo(['AAPL','MCD'], start = '2020-01-01', end = '2020-12-31')

X_levels_df.round(2).head()
X_df = X_levels_df['Adj Close'].pct_change()
X_df = X_df.dropna()

y_df = y_df.rename(columns={"^GSPC": "SP500"})

y_X_df = pd.concat([y_df, X_df], axis=1, join="inner")

#%% Data analysis

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(X_df["AAPL"], X_df["MCD"], y_df,
           s = 8, alpha = 0.5)

ax.set_xlabel('AAPL')
ax.set_ylabel('MCD')
ax.set_zlabel('SP500')
ax.set_xlim([-0.15,0.15])
ax.set_ylim([-0.15,0.15])
ax.set_zlim([-0.15,0.15])

g = sns.pairplot(y_X_df)
g.map_upper(sns.scatterplot, color = 'b')
g.map_lower(sns.kdeplot, levels=8, fill=True, cmap="viridis_r") 
g.map_diag(sns.distplot, kde=False, color = 'b')

#%% covariance matrix

SIGMA = y_X_df.cov()

fig, axs = plt.subplots()

h = sns.heatmap(SIGMA, annot=True,cmap='RdBu_r')
h.set_aspect("equal")

#%% correlation matrix

RHO = y_X_df.corr()

fig, axs = plt.subplots()

h = sns.heatmap(RHO, annot=True,cmap='RdBu_r')
h.set_aspect("equal")

#%% Volatility vector space

Angles = np.arccos(RHO)*180/np.pi
fig, axs = plt.subplots()

h = sns.heatmap(Angles, annot=True,cmap='RdBu_r')
h.set_aspect("equal")


#%% Regression

# add a column of ones
X_df = sm.add_constant(X_df)

model = sm.OLS(y_df, X_df)
results = model.fit()
print(results.summary())

p = model.fit().params
print(p)

# generate x-values for your regression line (two is sufficient)
xx1,xx2 = np.meshgrid(np.linspace(-0.15,0.15,20), np.linspace(-0.15,0.15,20))

yy = p.AAPL*xx1 + p.MCD*xx2 + p.const

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(X_df["AAPL"], X_df["MCD"], y_df,
           s = 8, alpha = 0.5)
ax.plot_wireframe(xx1, xx2, yy, color = 'r')
ax.set_xlim([-0.15,0.15])
ax.set_ylim([-0.15,0.15])
ax.set_zlim([-0.15,0.15])
ax.set_xlabel('AAPL')
ax.set_ylabel('MCD')
ax.set_zlabel('SP500')
