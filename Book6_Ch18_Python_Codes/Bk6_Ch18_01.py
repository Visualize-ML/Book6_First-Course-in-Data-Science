

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# single variate TLS regression

# initializations and download results 
import pandas as pd
import pandas_datareader as web
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas_datareader as web
import statsmodels.api as sm


y_levels_df = web.get_data_yahoo(['AAPL'], start = '2020-01-01', end = '2020-12-31')

y_levels_df.round(2).head()
y_df = y_levels_df['Adj Close'].pct_change()
y_df = y_df.dropna()


x_levels_df = web.get_data_yahoo(['^GSPC'], start = '2020-01-01', end = '2020-12-31')

x_levels_df.round(2).head()
x_df = x_levels_df['Adj Close'].pct_change()
x_df = x_df.dropna()

x_df = x_df.rename(columns={"^GSPC": "SP500"})

x_y_df = pd.concat([x_df, y_df], axis=1, join="inner")

#%% USE ODR in SciPy

from scipy.odr import *

# Define a function to fit the data with
def linear_func(b, x):
   b0, b1 = b
   return b1*x + b0

# Create a model for fitting
linear_model = Model(linear_func)

# Load data to the model
data = RealData(x_df.T, y_df.T)

# Set up ODR with the model and data
odr = ODR(data, linear_model, beta0=[0., 1.])

# Solve the regression
out = odr.run()

# Use the in-built pprint method to display results
out.pprint()

#%%

#%% TLS, matrix computation
import statsmodels.api as sm

SIMGA = x_y_df.cov()

Lambda, V = np.linalg.eig(SIMGA)

idx = Lambda.argsort()[::-1]   
Lambda = Lambda[idx]
V = V[:,idx]

lambda_min = np.min(Lambda)
b1_TLS = -V[0, 1]/V[1, 1]
print(b1_TLS)

b0_TLS = y_df.mean().values - b1_TLS*x_df.mean().values
print(b0_TLS)

#%% OLS regression

# add a column of ones
X_df = sm.add_constant(x_df)

model = sm.OLS(y_df, X_df)
results = model.fit()
print(results.summary())

p = model.fit().params

#%% visualization

b0 = out.beta[0]
b1 = out.beta[1]

# generate x-values for  regression line
x_ = np.linspace(x_df.min(),x_df.max(),10)

fig, ax = plt.subplots()

# scatter-plot data
plt.scatter(x_df, y_df, alpha = 0.5, 
            s = 8,label = 'Data')

plt.plot(x_, p.const + p.SP500 * x_,
         color = 'r', label = 'OLS')

plt.plot(x_, b0 + b1 * x_,
         color = 'b', label = 'TLS')

plt.axhline(y=0, color='k', linestyle='--')
plt.axvline(x=0, color='k', linestyle='--')
plt.axis('scaled')
plt.legend(loc='lower right')

plt.axis('scaled')
plt.ylabel('AAPL daily return')
plt.xlabel('S&P 500 daily return, market')
plt.xlim([-0.15,0.15])
plt.ylim([-0.15,0.15])

