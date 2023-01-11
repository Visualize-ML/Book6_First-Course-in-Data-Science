

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

from scipy.interpolate import lagrange
import matplotlib.pyplot as plt
import numpy as np

x_known = np.linspace(0, 6, num=7, endpoint=True)
y_known = np.sin(x_known)
# y_known = np.array([-1, -1, -1, 0, 1, 1, 1]) 

x_fine  = np.linspace(0, 6, num=300, endpoint=True)
y_fine  = np.sin(x_fine)

x = np.array([0, 1, 2])
y = x**3
poly = lagrange(x_known, y_known)


from numpy.polynomial.polynomial import Polynomial
Polynomial(poly).coef

fig, axs = plt.subplots()
plt.plot(x_known, y_known, 'or')
plt.plot(x_fine,  y_fine, 'r--',  linewidth = 0.25)
plt.plot(x_fine,  poly(x_fine), linewidth = 1.5)

for xc in x_known:
    plt.axvline(x=xc, color = [0.6, 0.6, 0.6], linewidth = 0.25)

plt.axhline(y=0, color = 'k', linewidth = 0.25)
plt.autoscale(enable=True, axis='x', tight=True)
plt.autoscale(enable=True, axis='y', tight=True)
plt.xlabel('x'); plt.ylabel('y')
plt.ylim([-1.1,1.1])
