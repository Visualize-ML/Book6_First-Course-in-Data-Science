

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def surface(x1,x2):

    v = (x1 + x2)*np.exp(-(x1**2 + x2**2))
    return v


X_scatter = np.random.uniform(-1,1,(25,2))
yy_scatter = surface(X_scatter[:,0],X_scatter[:,1])

x1_grid = np.linspace(-1, 1, 100)
x2_grid = np.linspace(-1, 1, 100)
xx1_grid, xx2_grid = np.meshgrid(x1_grid, x2_grid)

methods = ['nearest','linear', 'cubic']

for method in methods:
    
    yy_interp_2D = griddata(X_scatter, yy_scatter, (xx1_grid, xx2_grid), method=method)
    
    plt.figure()
    
    lims = dict(cmap='RdBu_r', vmin=-0.4, vmax=0.4)
    plt.pcolormesh(xx1_grid, xx2_grid, yy_interp_2D, shading='flat', **lims)
    plt.scatter(X_scatter[:,0],X_scatter[:,1], marker = 'x', c = 'k')
    plt.axis('scaled')
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.autoscale(enable=True, axis='y', tight=True)
    
    fig = plt.figure()
    ax = plt.axes(projection ="3d")
    
    ax.scatter(X_scatter[:,0],X_scatter[:,1], yy_scatter, marker = 'x', c = 'k')
    ax.plot_wireframe(xx1_grid, xx2_grid, yy_interp_2D)
