

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


X_df_no_1 = y_X_df[tickers[1:]]
y_df = y_X_df["SP500"]

labels = ['SP500','TSLA','WMT','MCD','USB',
           'YUM','NFLX','JPM','PFE',
           'F','GM','COST','JNJ'];


#%% Regression

import statsmodels.api as sm

# add a column of ones
X_df = sm.add_constant(X_df_no_1)

model = sm.OLS(y_df, X_df)
results = model.fit()
print(results.summary())

#%% Regression analysis

y = np.matrix(y_df.values).T
X = X_df.values
H = X@np.linalg.inv(X.T@X)@X.T

# coefficients
b = np.linalg.inv(X.T@X)@X.T@y

y_hat = H@y
e = y - y_hat

#%% Analysis of Variance

n = y.shape[0]
k = X.shape[1]
D = k - 1

I = np.identity(n)
J = np.ones((n,n))
vec_1 = np.ones_like(y)

y_bar = vec_1.T@y/n

# Sum of Squares for Total, SST
SST = y.T@(I - J/n)@y
MST = SST/(n - 1)
MST = MST[0,0]

#%% Sum of Squares for Error, SSE

SSE = y.T@(I - H)@y

# mean squared error, MSE
MSE = SSE/(n - k)
MSE_ = e.T@e/(n - k)
MSE = MSE[0,0]

#%% Sum of Squares for Regression, SSR

SSR = y.T@(H - J/n)@y
MSR = SSR/D
MSR = MSR[0,0]

#%% Orthogonal relationships

print('SST = ',SST)
print('SSR + SSE = ',SSR + SSE)

print('================')
print('y.T@y = ',y.T@y)
print('y_hat.T@y_hat + e.T@e = ',y_hat.T@y_hat + e.T@e)

print('================')
print('e.T@vec_1 = ', e.T@vec_1)

print('================')
print('e.T@(y_hat - y_bar*vec_1) = ', e.T@(y_hat - y_bar))

print('================')
print('e.T@(y - y_bar*vec_1) = ', e.T@(y - y_bar))

print('================')
print('e.T@X = ', e.T@X)

print('================')
print('e.T@X@b = ', e.T@X@b)


#%% R squared goodness of fit

R_squared = SSR/SST
R_sqaured_adj = 1 - MSE/MST

#%% F test

F = MSR/MSE

from scipy import stats
p_value_F = 1.0 - stats.f.cdf(F,k - 1,n - k)

#%% Log-likelihood

sigma_MLE = np.sqrt(SSE/n)

ln_L = -n*np.log(sigma_MLE*np.sqrt(2*np.pi)) - SSE/2/sigma_MLE**2

AIC = 2*k - 2*ln_L
BIC = k*np.log(n) - 2*ln_L


#%% t test

C = MSE*np.linalg.inv(X.T@X)

SE_b = np.sqrt(np.diag(C))
SE_b = np.matrix(SE_b).T

T = b/SE_b
p_one_side = 1 - stats.t(n - k).cdf(np.abs(T))
p = p_one_side*2
# P > |t|

#%% confidence interval of coefficients, 95%

alpha = 0.05
t = stats.t(n - k).ppf(1 - alpha/2)
b_lower_CI = b - t*SE_b # 0.025
b_upper_CI = b + t*SE_b # 0.975

#%% multi-collinearity

print('Rank of X')
print(np.linalg.matrix_rank(X))

print('det(X.T@X)')
print(np.linalg.det(X.T@X))

from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF

VIF_X_no_1_df = pd.Series([VIF(X_df.values, i) 
                           for i in range(X_df.shape[1])], 
                          index=X_df.columns)

print(VIF_X_no_1_df)

VIF_X_df = pd.Series([VIF(X_df_no_1.values, i) 
               for i in range(X_df_no_1.shape[1])], 
              index=X_df_no_1.columns)

print(VIF_X_df)

#%% Conditional probability

# covariance matrix
SIGMA_df = y_X_df.cov()
SIGMA = SIGMA_df.to_numpy()

fig, axs = plt.subplots()

h = sns.heatmap(SIGMA_df,cmap='RdBu_r', linewidths=.05)
h.set_aspect("equal")
h.set_title('$\Sigma$')

# blocks
SIGMA_Xy = np.matrix(SIGMA[1:,0]).T
SIGMA_XX = np.matrix(SIGMA[1:,1:])

SIGMA_XX_inv = np.linalg.inv(SIGMA_XX)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

plt.sca(axs[0])
ax = sns.heatmap(SIGMA_XX,cmap='RdBu_r', cbar=False, 
                 xticklabels = labels[1:],
                 yticklabels = labels[1:], 
                 linewidths=.05)

ax.set_aspect("equal")
plt.title('$\Sigma_{XX}$')

plt.sca(axs[1])
ax = sns.heatmap(SIGMA_XX_inv,cmap='RdBu_r', cbar=False,
                 xticklabels = labels[1:],
                 yticklabels = labels[1:], 
                 linewidths=.05)

ax.set_aspect("equal")
plt.title('$\Sigma_{XX}^{-1}$')

# calculate coefficient vector, b

b = SIGMA_XX_inv@SIGMA_Xy

fig, axs = plt.subplots(1, 5, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(b,cmap='RdBu_r', cbar=False,
                 yticklabels = labels[1:], 
                 linewidths=.05)

ax.set_aspect("equal")
plt.title('$b$')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(SIGMA_XX_inv,cmap='RdBu_r', cbar=False,
                 xticklabels = labels[1:],
                 yticklabels = labels[1:], 
                 linewidths=.05)

ax.set_aspect("equal")
plt.title('$\Sigma_{XX}^{-1}$')

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(SIGMA_Xy,cmap='RdBu_r', cbar=False,
                 xticklabels = [labels[0]],
                 yticklabels = labels[1:], 
                 linewidths=.05)

ax.set_aspect("equal")
plt.title('$\Sigma_{Xy}$')

#%% calculate coefficient, b0

MU = y_X_df.mean()
MU = np.matrix(MU.to_numpy())

b0 = MU[0,0] - MU[0,1:]@b

fig, axs = plt.subplots(1, 7, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(b0,cmap='RdBu_r', cbar=False, 
                 linewidths=.05,xticklabels = [],
                 yticklabels = [])

ax.set_aspect("equal")
plt.title('$b_0$')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(np.matrix(MU[0,0]),cmap='RdBu_r', cbar=False,
                 linewidths=.05, xticklabels = [],
                 yticklabels = [])

ax.set_aspect("equal")
plt.title('$\mu_{y}$')

plt.sca(axs[3])
plt.title('-')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(MU[0,1:],cmap='RdBu_r', cbar=False,
                 xticklabels = labels[1:], yticklabels = [],
                 linewidths=.05)
ax.set_aspect("equal")
plt.title('$\mu_{X}$')


plt.sca(axs[5])
plt.title('@')
plt.axis('off')

plt.sca(axs[6])
ax = sns.heatmap(b,cmap='RdBu_r', cbar=False,
                 yticklabels = labels[1:], xticklabels = [],
                 linewidths=.05)

ax.set_aspect("equal")
plt.title('$b$')
