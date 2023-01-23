

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

X_y_df = pd.concat([X_df, y_df], axis=1, join="inner")

#%% USE ODR in scipy

from scipy.odr import *

# Define a function to fit data
def linear_func(b, x):
   # b0, b1, b2 = b
   # x1, x2 = x
   # return b2*x2 + b1*x1 + b0
   b0 = b[0]
   b_ = b[1:]
   return b_.T@x + b0 

# Create a model for fitting
linear_model = Model(linear_func)

# Create a RealData object using our initiated data
data = RealData(X_df.T, y_df.T)

# Set up ODR with the model and data
odr = ODR(data, linear_model, beta0=[0., 1., 1])

# Run the regression
out = odr.run()

# Use pprint method to display results
out.pprint()


#%% TLS, matrix computation

SIMGA = X_y_df.cov()

Lambda, V = np.linalg.eig(SIMGA)

idx = Lambda.argsort()[::-1]   
Lambda = Lambda[idx]
V = V[:,idx]

lambda_min = np.min(Lambda)

b1_TLS_ = -V[0,2]/V[2,2]
b2_TLS_ = -V[1,2]/V[2,2]

print(b1_TLS_)
print(b2_TLS_)

b0_TLS_ = y_df.mean().values - [b1_TLS_, b2_TLS_]@X_df.mean().values
print(b0_TLS_)

b0_TLS = out.beta[0]
b1_TLS = out.beta[1]
b2_TLS = out.beta[2]

#%% OLS Regression

# add a column of ones
X_df = sm.add_constant(X_df)

model = sm.OLS(y_df, X_df)
results = model.fit()
print(results.summary())

p = model.fit().params
print(p)

# generate x-values for your regression line (two is sufficient)
xx1,xx2 = np.meshgrid(np.linspace(-0.15,0.15,20), np.linspace(-0.15,0.15,20))

yy_OLS = p.AAPL*xx1 + p.MCD*xx2 + p.const
yy_TLS = b1_TLS*xx1 + b2_TLS*xx2 + b0_TLS

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(X_df["AAPL"], X_df["MCD"], y_df,
           s = 8, alpha = 0.5)

ax.plot_wireframe(xx1, xx2, yy_OLS, color = 'r', label = 'OLS')
ax.plot_wireframe(xx1, xx2, yy_TLS, color = 'b', label = 'TLS')

ax.set_xlim([-0.15,0.15])
ax.set_ylim([-0.15,0.15])
ax.set_zlim([-0.15,0.15])
ax.set_xlabel('AAPL')
ax.set_ylabel('MCD')
ax.set_zlabel('SP500')

plt.legend(loc='lower right')
