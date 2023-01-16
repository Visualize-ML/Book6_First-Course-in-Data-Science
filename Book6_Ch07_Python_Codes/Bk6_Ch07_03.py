

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
tickers = ['^GSPC','TSLA'];
df = web.get_data_yahoo(tickers, start = '2015-08-01', end = '2021-08-01')

df = df.dropna()

#%% daily log return

daily_log_r = df['Adj Close'].apply(lambda x: np.log(x) - np.log(x.shift(1)))
daily_log_r = daily_log_r.dropna()

df_corr = daily_log_r['^GSPC'].rolling(100).corr(daily_log_r['TSLA'])

fig, ax = plt.subplots()

# daily return of selected date range
ax.plot(df_corr[df_corr.first_valid_index():df_corr.index[-1]])
ax.axhline(y=0.5, color='r', linestyle='-')
ax.set_ylabel('Rolling correlation')
