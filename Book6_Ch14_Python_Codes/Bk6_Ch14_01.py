

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

clf = Ridge()

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

#%% OLS

import statsmodels.api as sm

# add a column of ones
X_df = sm.add_constant(X_df)

model = sm.OLS(y_df, X_df)
results = model.fit()
print(results.summary())

b = model.fit().params
b = b.values

#%% Ridge regression

coefs = []
errors = []
coeff_df = pd.DataFrame()

alphas = np.logspace(-4, 2, 200)

# Train the model with different regularisation strengths
for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(X_df, y_df)
    coefs.append(clf.coef_)
    errors.append(mean_squared_error(clf.coef_, b))
    
    b_i = clf.coef_
    b_X_df = pd.DataFrame(data=b_i[1:].T, index = tickers[1:], columns=[a])
    
    coeff_df = pd.concat([coeff_df, b_X_df], axis = 1)

fig, ax = plt.subplots()
h = sns.lineplot(data=coeff_df.T,markers=False, dashes=False,palette = "husl")
plt.axhline(y=0, color='k', linestyle='--')
h.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
ax.set_xscale('log')
# ax.grid(which='minor', axis='x', linestyle='--')

fig, ax = plt.subplots()

ax.plot(alphas, errors)
plt.fill_between(alphas,errors, color = '#DEEAF6')
ax.set_xscale('log')
plt.xlabel('$\u03B1$')
plt.ylabel('Coefficient error')
plt.axis('tight')

plt.show()
