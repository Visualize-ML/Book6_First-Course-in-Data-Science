

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


import numpy as np
import matplotlib.pyplot as plt

lambdas = np.linspace(0.9,0.99,10)
i_day   = np.linspace(1,20,20)

lambda_l, ii = np.meshgrid(lambdas,i_day)

ww = (1 - lambda_l)*lambda_l**(ii - 1)

fig, ax = plt.subplots()

colors = plt.cm.jet(np.linspace(0,1,10))

for i in np.linspace(1,10,10):
    plt.plot(i_day,ww[:,int(i)-1],marker = 'x',
             color = colors[int(i)-1],
             label = '$\lambda = {lll:.2f}$'.format(lll = lambdas[int(i)-1]))

plt.xlabel('Day, i')
plt.ylabel('EWMA weight')
plt.xticks(i_day)
plt.legend()
ax.invert_xaxis()

HL = np.log(0.5)/np.log(lambdas)

fig, ax = plt.subplots()

plt.plot(lambdas,HL,marker = 'x')
plt.xlabel('$\lambda$')
plt.ylabel('Half life')
