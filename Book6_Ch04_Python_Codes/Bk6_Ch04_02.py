

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
  
# original data: exponential distribution
original_X = np.random.exponential(size = 1000)
  
# Box-Cox tpower transformation
new_X, fitted_lambda = stats.boxcox(original_X)

# Yeo-Johnson power transformation
# new_X, fitted_lambda = stats.yeojohnson(original_X)


fig, ax = plt.subplots(1, 2)

sns.distplot(original_X, hist = True, 
             kde = True,
             label = "Original", ax = ax[0])
  
sns.distplot(new_X, hist = True, 
             kde = True,
             label = "Original", ax = ax[1])

# QQ plot

fig, ax = plt.subplots(1, 2)

stats.probplot(original_X, dist=stats.norm, plot=ax[0])
ax[0].set_xlabel('Normal')
ax[0].set_ylabel('Original data')
ax[0].set_title('')

stats.probplot(new_X, dist=stats.norm, plot=ax[1])
ax[1].set_xlabel('Normal')
ax[1].set_ylabel('Transformed data')
ax[1].set_title('')
