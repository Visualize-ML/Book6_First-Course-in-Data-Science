

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

def surface(x1,x2):

    v = (x1 + x2)*np.exp(-2*(x1**2 + x2**2))
    return v

x1_data = np.linspace(-1, 1, 5)
x2_data = np.linspace(-1, 1, 5)
xx1_data, xx2_data = np.meshgrid(x1_data, x2_data)
yy_data = surface(xx1_data,xx2_data)


x1_grid = np.linspace(-1.1, 1.1, 23)
x2_grid = np.linspace(-1.1, 1.1, 23)
xx1_grid, xx2_grid = np.meshgrid(x1_grid, x2_grid)


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = plt.axes(projection ="3d")

ax.scatter(xx1_data, xx2_data, yy_data, marker = 'x', c = 'k')
ax.plot_wireframe(xx1_data, xx2_data, yy_data)

methods = ['linear', 'cubic']

for kind in methods:
    
    f_interp = interp2d(xx1_data, xx2_data, yy_data, kind=kind)
    yy_interp_2D = f_interp(x1_grid, x2_grid)
    
    plt.figure()
    
    lims = dict(cmap='RdBu_r', vmin=-0.4, vmax=0.4)
    plt.pcolormesh(xx1_grid, xx2_grid, yy_interp_2D, shading='flat', **lims)
    plt.scatter(xx1_data.ravel(),xx2_data.ravel(), marker = 'x', c = 'k')
    plt.axis('scaled')
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.autoscale(enable=True, axis='y', tight=True)
    
    fig = plt.figure()
    ax = plt.axes(projection ="3d")
    
    ax.scatter(xx1_data, xx2_data, yy_data, marker = 'x', c = 'k')
    ax.plot_wireframe(xx1_grid, xx2_grid, yy_interp_2D)
