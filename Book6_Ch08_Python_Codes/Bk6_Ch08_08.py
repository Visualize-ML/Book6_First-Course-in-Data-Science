

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# initializations and download results 
import pandas as pd
import pandas_datareader as web
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas_datareader as web
tickers = ['TSLA','TSM','COST','NVDA','FB','AMZN','AAPL','NFLX','GOOGL'];
stock_levels_df = web.get_data_yahoo(tickers, start = '2020-08-01', end = '2021-08-01')
stock_levels_df.to_csv("9_stocks_level.csv")

print(stock_levels_df.round(2).head())
print(stock_levels_df.round(2).tail())

#%% Plot lineplot of stock prices


fig = sns.relplot(data=stock_levels_df['Adj Close'],dashes = False,
            kind="line") # , palette="coolwarm"
fig.set_axis_labels('Date','Adjusted closing price')


# normalize the initial stock price levels to 1
normalized_stock_levels = stock_levels_df['Adj Close']/stock_levels_df['Adj Close'].iloc[0]

fig = sns.relplot(data=normalized_stock_levels,dashes = False,
            kind="line") # , palette="coolwarm"
fig.set_axis_labels('Date','Normalized closing price')

#%% daily log return

daily_log_r = stock_levels_df['Adj Close'].apply(lambda x: np.log(x) - np.log(x.shift(1)))

daily_log_r = daily_log_r.dropna()

#%% Variance-covariance matrix
# Compute the covariance matrix
cov_SIGMA = daily_log_r.cov()

# Set up the matplotlib figure
fig, ax = plt.subplots()

sns.heatmap(cov_SIGMA, cmap="coolwarm",
            square=True, linewidths=.05)
plt.title('Covariance matrix of historical data')

#%% correlation matrix

# Compute the correlation matrix
corr_P = daily_log_r.corr()

# Set up the matplotlib figure
fig, ax = plt.subplots()

sns.heatmap(corr_P, cmap="coolwarm",
            square=True, linewidths=.05, annot=True,
            vmax = 1,vmin = 0)
plt.title('Correlation matrix of historical data')

#%% Cholesky decomposition

import scipy.linalg


L = scipy.linalg.cholesky(cov_SIGMA, lower=True)
R = scipy.linalg.cholesky(cov_SIGMA, lower=False)

fig, axs = plt.subplots(1, 5, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(cov_SIGMA,cmap='coolwarm', cbar=False)
ax.set_aspect("equal")
plt.title('$\Sigma$')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(L,cmap='coolwarm',
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('L')

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(R,cmap='coolwarm',
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('R')

#%% Correlated stock price simulation

# convert daily parameters to yearly
mu_yearly = daily_log_r.mean()*250
R_yearly  = R*np.sqrt(250)
SIGMA_yearly = cov_SIGMA*250

n = 250
# simulation steps

dt = 1/250 
# assume 250 business days in a year

S0 = stock_levels_df['Adj Close'].iloc[-1]

S0 = np.array(S0)
# current stock price levels

Z = np.random.normal(0, 1, size=(n, 9))
# only simulate one set of paths

drift = (mu_yearly - np.diag(SIGMA_yearly)/2)*dt;
drift = np.array(drift)

vol  = Z@R_yearly*np.sqrt(dt);

S = np.exp(drift + vol)

S = np.vstack([np.ones(9), S])
# add a layer of ones

S = S0 * S.cumprod(axis=0)
# compute the stock levels

Sim_df = pd.DataFrame(data=S, columns=tickers)
# convert the result to a dataframe

fig = sns.relplot(data=Sim_df,dashes = False,
            kind="line") # , palette="coolwarm"

plt.xlabel("$t$")
plt.ylabel("$S$")
plt.title('Simulated levels, one set of paths')

#%% Compute the correlation matrix of the simulated results

daily_log_sim = Sim_df.apply(lambda x: np.log(x) - np.log(x.shift(1)))
daily_log_sim = daily_log_sim.dropna()


# Compute the correlation matrix
corr_P_sim = daily_log_sim.corr()

# Set up the matplotlib figure
fig, ax = plt.subplots()

sns.heatmap(corr_P_sim, cmap="coolwarm",
            square=True, linewidths=.05, annot=True,
            vmax = 1,vmin = 0)

plt.title('Correlation matrix of simulated results')

# calculate the differences between historical and simulated
fig, ax = plt.subplots()

sns.heatmap(corr_P - corr_P_sim, cmap="coolwarm",
            square=True, linewidths=.05, annot=True,
            vmax = 1,vmin = 0)

plt.title('Differences, correlation matrix')
