

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer

np.random.seed(123)

def func_exp(x, a, b, c):
    # exponential function
    
    return a * np.exp(b * x) + c


def func_log(x, a, b, c):
    # log function
    
    return a * np.log(b * x) + c


def generate_data(func, *args, noise=0):
    
    # generate data
    xs = np.linspace(1, 6, 50)
    ys = func(xs, *args)
    noise = noise * np.random.normal(size=len(xs)) + noise
    xs = xs.reshape(-1, 1) 
    ys = (ys + noise).reshape(-1, 1)
    return xs, ys

#%% Fit exponential data

# Generate data
x_samp, y_samp = generate_data(func_exp, 2.5, 1.2, 0.7, noise=10)

transformer = FunctionTransformer(np.log, validate=True)

y_trans = transformer.fit_transform(y_samp)

# Regression
regressor = LinearRegression()
results = regressor.fit(x_samp, y_trans)
model = results.predict
y_fit = model(x_samp)


plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.scatter(x_samp, y_samp)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.tight_layout()

plt.subplot(122)
plt.scatter(x_samp, y_samp)
plt.yscale('log')
plt.ylabel('ln(y)')
plt.xlabel('x')
plt.grid(True)
plt.tight_layout()

#%% fitted data

plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.scatter(x_samp, y_samp, label = 'Data')
plt.plot(x_samp, np.exp(y_fit), "r", label="Fitted")
plt.legend()
plt.ylabel('y')
plt.xlabel('x')
plt.grid(True)
plt.tight_layout()

plt.subplot(122)
plt.scatter(x_samp, y_samp, label = 'Data')
plt.plot(x_samp, np.exp(y_fit), "r", label="Fitted")
plt.yscale('log')
plt.ylabel('ln(y)')
plt.xlabel('x')
plt.legend()
plt.grid(True)
plt.tight_layout()

#%% Fit log data

# Data
x_samp, y_samp = generate_data(func_log, 2.5, 1.2, 0.7, noise=0.3)
x_trans = transformer.fit_transform(x_samp)

# Regression
regressor = LinearRegression()
results = regressor.fit(x_trans, y_samp)
model = results.predict
y_fit = model(x_trans)

plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.scatter(x_samp, y_samp)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.tight_layout()

plt.subplot(122)
plt.scatter(x_samp, y_samp)
plt.xscale('log')
plt.xlabel('ln(x)')
plt.ylabel('y')
plt.grid(True)
plt.tight_layout()

#%% fitted data

plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.scatter(x_samp, y_samp, label = 'Data')
plt.plot(x_samp, y_fit, "r", label="Fitted")
plt.legend()
plt.ylabel('y')
plt.xlabel('x')
plt.grid(True)
plt.tight_layout()

plt.subplot(122)
plt.scatter(x_samp, y_samp, label = 'Data')
plt.plot(x_samp, y_fit, "r", label="Fitted")
plt.xscale('log')
plt.xlabel('ln(x)')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.tight_layout()
