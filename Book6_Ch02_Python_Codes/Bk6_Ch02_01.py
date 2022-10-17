

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import seaborn as sns

X, y = load_iris(as_frame=True, return_X_y=True)
X.head()

iris_df = X.copy()
iris_df['species'] = y
sns.pairplot(iris_df, hue='species', palette = "bright")

#%% Generate X with missing values

X_NaN = X.copy()

mask = np.random.uniform(0,1,size = X_NaN.shape)

mask = (mask <= 0.4)

X_NaN[mask] = np.NaN
print(X_NaN.tail)

import seaborn as sns
iris_df_NaN = X_NaN.copy()
iris_df_NaN['species'] = y
sns.pairplot(iris_df_NaN, hue='species', palette = "bright")

#%% Visualize missing values

is_NaN = iris_df_NaN.isna()
print(is_NaN)

fig, ax = plt.subplots()
ax = sns.heatmap(is_NaN,
                 cmap='gray_r',
                 cbar=False)

not_NaN = iris_df_NaN.notna()
# sum_rows = not_NaN. sum(axis=1)
print(not_NaN)

fig, ax = plt.subplots()
ax = sns.heatmap(not_NaN,
                 cmap='gray_r',
                 cbar=False)

print("\nCount total NaN at each column:\n",
      X_NaN.isnull().sum())

print("\nPercentage of NaN at each column:\n",
      X_NaN.isnull().sum()/len(X_NaN)*100)

import missingno as msno
# missingno has to be installed first

msno.matrix(iris_df_NaN)


#%% drop missing value rows

X_NaN_drop = X_NaN.dropna(axis=0)

iris_df_NaN_drop = pd.DataFrame(X_NaN_drop, columns=X_NaN.columns, index=X_NaN.index)
iris_df_NaN_drop['species'] = y
sns.pairplot(iris_df_NaN_drop, hue='species', palette = "bright")

#%% imputing the data using median imputation

from sklearn.impute import SimpleImputer

# The imputation strategy:
# 'mean', replace missing values using the mean along each column
# 'median', replace missing values using the median along each column
# 'most_frequent', replace missing using the most frequent value along each column
# 'constant', replace missing values with fill_value

si = SimpleImputer(strategy='median')
# impute training data
X_NaN_median = si.fit_transform(X_NaN)

iris_df_NaN_median = pd.DataFrame(X_NaN_median, columns=X_NaN.columns, index=X_NaN.index)
iris_df_NaN_median['species'] = y
sns.pairplot(iris_df_NaN_median, hue='species', palette = "bright")

#%% kNN imputation
# kNN, k nearest neighbours

from sklearn.impute import KNNImputer

knni = KNNImputer(n_neighbors=5) 
X_NaN_kNN = knni.fit_transform(X_NaN)

iris_df_NaN_kNN = pd.DataFrame(X_NaN_kNN, columns=X_NaN.columns, index=X_NaN.index)
iris_df_NaN_kNN['species'] = y

sns.pairplot(iris_df_NaN_kNN, hue='species', palette = "bright")

#%% iterative imputer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.ensemble import RandomForestRegressor

rf_imp = IterativeImputer(estimator=RandomForestRegressor(random_state=0), max_iter=20)
X_NaN_RF = rf_imp.fit_transform(X_NaN)


iris_df_NaN_RF = pd.DataFrame(X_NaN_RF, columns=X_NaN.columns, index=X_NaN.index)
iris_df_NaN_RF['species'] = y
sns.pairplot(iris_df_NaN_RF, hue='species', palette = "bright")


