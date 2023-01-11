

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# multi-variate regression

# initializations
import pandas as pd
import pandas_datareader as web
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

X_tickers = ['TSLA','WMT','MCD','USB',
           'YUM','NFLX','JPM','PFE',
           'F','GM','COST','JNJ'];

y_levels_df = web.get_data_yahoo(['^GSPC'], start = '2020-01-01', end = '2020-12-31')

y_levels_df.round(2).head()
y_df = y_levels_df['Adj Close'].pct_change()
y_df = y_df.dropna()

X_levels_df = web.get_data_yahoo(X_tickers, start = '2020-01-01', end = '2020-12-31')

X_levels_df.round(2).head()
X_df = X_levels_df['Adj Close'].pct_change()
X_df = X_df.dropna()

y_df = y_df.rename(columns={"^GSPC": "SP500"})

X_y_df = pd.concat([X_df, y_df], axis=1, join="inner")

#%% TLS, matrix computation

SIMGA = X_y_df.cov()

Lambda, V = np.linalg.eig(SIMGA)

idx = Lambda.argsort()[::-1]   
Lambda = Lambda[idx]
V = V[:,idx]

lambda_min = np.min(Lambda)

D = len(X_tickers)

b_TLS_ = -V[0:D,D]/V[D,D]

print(b_TLS_)

b0_TLS_ = y_df.mean().values - b_TLS_@X_df.mean().values
print(b0_TLS_)

b_TLS = np.hstack((b0_TLS_,b_TLS_))

labels = ['const'] + X_tickers
b_df_TLS = pd.DataFrame(data=b_TLS.T, index=[labels], columns=['TLS']) 

#%% OLS Regression
import statsmodels.api as sm

# add a column of ones
X_df = sm.add_constant(X_df)

model = sm.OLS(y_df, X_df)
results = model.fit()
print(results.summary())

b_df_OLS = model.fit().params
print(b_df_OLS)

b_df_OLS = pd.DataFrame(data=b_df_OLS.values, index=[labels], columns=['OLS']) 


coeffs = pd.concat([b_df_TLS, b_df_OLS], axis=1, join="inner")

fig, ax = plt.subplots()
coeffs.plot.bar()
# h.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.axhline(y=0, color='r', linestyle='--')
