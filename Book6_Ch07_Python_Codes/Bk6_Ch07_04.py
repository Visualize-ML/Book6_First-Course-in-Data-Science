

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############



# Rolling regression

import pandas as pd
import pandas_datareader as web
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas_datareader as web
import statsmodels.api as sm

y_levels_df = web.get_data_yahoo(['TSLA'], start='08-01-2018', end='08-01-2021')

y_levels_df.round(2).head()
y_df = y_levels_df['Adj Close'].pct_change()
y_df = y_df.dropna()


x_levels_df = web.get_data_yahoo(['^GSPC'], start='08-01-2018', end='08-01-2021')

x_levels_df.round(2).head()
x_df = x_levels_df['Adj Close'].pct_change()
x_df = x_df.dropna()

x_df = x_df.rename(columns={"^GSPC": "SP500"})

#%% rolling regression 

from statsmodels.regression.rolling import RollingOLS

# add a column of ones
X_df = sm.add_constant(x_df)

rols = RollingOLS(y_df, X_df, window=100)
rres = rols.fit()
params = rres.params
print(params.head())
print(params.tail())

#%% Visualization

fig = rres.plot_recursive_coefficient(variables=['SP500'])
plt.ylabel('Coefficient')
plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
plt.axhline(y=0, color='r', linestyle='--')
plt.axhline(y=1, color='r', linestyle='--')
fig.tight_layout() 


fig = rres.plot_recursive_coefficient(variables=['const'])
plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
plt.axhline(y=0, color='r', linestyle='--')
plt.ylabel('Constant')
fig.tight_layout() 
