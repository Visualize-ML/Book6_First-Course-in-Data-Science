

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader 
import scipy.stats as stats
import pylab

df = pandas_datareader.data.DataReader(['UNRATENSA'], data_source='fred', start='08-01-1950', end='08-01-2021')
df = df.dropna()

#%% long term trend

plt.close('all')

average_rate = df['UNRATENSA'].mean()

fig, axs = plt.subplots()

df['UNRATENSA'].plot()
plt.xlabel('Date')
plt.ylabel('Unemployment rate')
plt.show()

plt.axhline(y=average_rate, color= 'r', zorder=0)
plt.axhline(y=df['UNRATENSA'].max(), color= 'r', zorder=0)
plt.axhline(y=df['UNRATENSA'].min(), color= 'r', zorder=0)
axs.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

#%% Zoom in

fig, axs = plt.subplots()

df['UNRATENSA']['1989-01-01':'1999-01-01'].plot()
plt.xlabel('Date')
plt.ylabel('Unemployment rate')
plt.show()

axs.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])


#%%
import seaborn as sns

df['year'] = pd.DatetimeIndex(df.index).year

df['month'] = pd.DatetimeIndex(df.index).month
import calendar
df['month'] = df['month'].apply(lambda x: calendar.month_abbr[x])

#%%
fig, axs = plt.subplots()

sns.lineplot(data=df['1989-01-01':'1999-01-01'], x="year", y="UNRATENSA", hue="month")

fig, axs = plt.subplots()

sns.lineplot(data=df['1989-01-01':'1999-01-01'], x="month", y="UNRATENSA", hue="year")

#%%

fig, axs = plt.subplots()

sns.boxplot(x='year', y='UNRATENSA', data=df)
plt.xticks(rotation = 90)


fig, axs = plt.subplots()

sns.boxplot(x='month', y='UNRATENSA', data=df)
plt.xticks(rotation = 45)
