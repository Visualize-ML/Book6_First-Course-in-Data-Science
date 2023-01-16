

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

import pandas_datareader 
import scipy.stats as stats
import pylab

df = pandas_datareader.data.DataReader(['sp500'], data_source='fred', start='01-01-2021', end='08-01-2021')

#%% Generate X with missing values

df_NaN = df.copy()

mask = np.random.uniform(0,1,size = df_NaN.shape)

mask = (mask <= 0.3)

df_NaN[mask] = np.NaN
print(df_NaN.tail)

#%%

import matplotlib.pyplot as plt

# ffill() is equivalent to fillna(method='ffill') and 
# bfill() is equivalent to fillna(method='bfill')

df_NaN_forward  = df_NaN.ffill()
df_NaN_backward = df_NaN.bfill()

fig, axs = plt.subplots()

df_NaN_forward['sp500'].plot(color = 'r')
df_NaN['sp500'].plot(marker = 'x')
plt.xlabel('Date')
plt.ylabel('Price level with NaN')
plt.show()

fig, axs = plt.subplots()

df_NaN_backward['sp500'].plot(color = 'r')
df_NaN['sp500'].plot(marker = 'x')
plt.xlabel('Date')
plt.ylabel('Price level with NaN')
plt.show()

#%% interpolation

# If you are dealing with a time series that is growing at an increasing rate, method='quadratic' may be appropriate.
# If you have values approximating a cumulative distribution function, then method='pchip' should work well.
# To fill missing values with goal of smooth plotting, consider method='akima'.

df_NaN_interpolate = df_NaN.interpolate()

fig, axs = plt.subplots()

df_NaN_interpolate['sp500'].plot(color = 'r')
df_NaN['sp500'].plot(marker = 'x')
plt.xlabel('Date')
plt.ylabel('Price level with NaN')
plt.show()
