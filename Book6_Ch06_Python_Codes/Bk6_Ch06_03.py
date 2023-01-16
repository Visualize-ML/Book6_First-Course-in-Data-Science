

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader 
import scipy.stats as stats

df = pandas_datareader.data.DataReader(['UNRATENSA'], data_source='fred', start='08-01-2000', end='08-01-2021')
df = df.dropna()

# deal with missing values
df['UNRATENSA'].interpolate(inplace=True)

res = sm.tsa.seasonal_decompose(df['UNRATENSA'])

# generate subplots
resplot = res.plot()

res.resid
res.seasonal
res.trend

#%% Original data

fig, axs = plt.subplots()

df['UNRATENSA'].plot()
plt.xlabel('Date')
plt.ylabel('Original')
plt.show()

#%% plot trend on top of original curve

df['UNRATENSA'].plot()
res.trend.plot(color = 'r')
plt.xlabel('Date')
plt.ylabel('Trend')
plt.show()

#%% plot seasonal component

fig, axs = plt.subplots()

res.seasonal.plot()
plt.axhline(y = 0, color = 'r')
plt.xlabel('Date')
plt.ylabel('Seasonal')
plt.show()

#%% plot irregular

fig, axs = plt.subplots()

res.resid.plot()
plt.axhline(y = 0, color = 'r')
plt.xlabel('Date')
plt.ylabel('irregular')
plt.show()
