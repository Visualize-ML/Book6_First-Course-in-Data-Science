

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

x = np.linspace(-5, 5, 100)

f_x = expit(x)

# Plot the logistic function

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(x, f_x, 'b-')
ax.set_xlabel('$x$')
ax.set_ylabel('$f(x)$')
ax.set_xlim(-5, 5)
plt.show()
