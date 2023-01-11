

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import matplotlib.pyplot as plt
import numpy as np

methods = ['none', 'nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos', 'blackman']

# Fixing random state for reproducibility
def surface(x1,x2):

    v = (x1 + x2)*np.exp(-2*(x1**2 + x2**2))
    return v

x1_data = np.linspace(-1, 1, 5)
x2_data = np.linspace(-1, 1, 5)
xx1_data, xx2_data = np.meshgrid(x1_data, x2_data)
yy_data = surface(xx1_data,xx2_data)

fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(9, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

for ax, interp_method in zip(axs.flat, methods):
    ax.imshow(yy_data, interpolation=interp_method, cmap='RdBu_r')
    ax.set_title(str(interp_method))

plt.tight_layout()
plt.show()

# reference: https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html
