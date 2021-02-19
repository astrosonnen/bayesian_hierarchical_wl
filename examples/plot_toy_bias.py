import pylab
import numpy as np
from matplotlib import rc
rc('text', usetex=True)


np.random.seed(0)

ngal = 500

fsize = 20

colors = 'b'

x = np.random.rand(ngal) + 10.5
#x = np.random.normal(10.5, 0.3, 1000)
x = x[x > 10.5]
ngal = len(x)

y0 = 0.5
beta = 2.
err = 0.1
y = y0 + beta*(x - 11.) + np.random.normal(0., 0.1, ngal)

x_obs = x + np.random.normal(0., err, ngal)

mbin = (x_obs > 10.9) & (x_obs < 11.1)

y_med = np.median(y[mbin])

#large = mbin & (y > y_med)
#small = mbin & (y <= y_med)

large = mbin & (y > y0 + beta*(x_obs - 11.))
small = mbin & (y <= y0 + beta*(x_obs - 11.))

print(x[small].mean(), x[large].mean())

nsmall = len(y[small])
nlarge = len(y[large])

ssize = 40
fig = pylab.figure()
pylab.subplots_adjust(left=0.08, bottom=0.08, right=0.99, top=0.9)

ax = fig.add_subplot(1, 1, 1)
pylab.title('Observed values', fontsize=fsize)
pylab.scatter(x_obs, y, color='gray', s=ssize)
pylab.scatter(x_obs[large], y[large], color='r', s=ssize, label='Above $M_*-R_e$ relation')
pylab.scatter(x_obs[small], y[small], color=colors, marker='o', s=ssize, label='Below $M_*-R_e$ relation')
pylab.axvline(10.9, color='k')
pylab.axvline(11.1, color='k')
pylab.xticks(())
pylab.yticks(())
pylab.xlabel('$\log{M_*}$', fontsize=fsize)
pylab.ylabel('$\log{R_e}$', fontsize=fsize)
pylab.xlim(10.5, 11.5)
pylab.legend(loc = 'upper left', fontsize=16, scatterpoints=1)
pylab.savefig('toy_bias_observed.png')
pylab.show()

fig = pylab.figure()
pylab.subplots_adjust(left=0.08, bottom=0.08, right=0.99, top=0.9)

ax = fig.add_subplot(1, 1, 1)
pylab.title('True values', fontsize=fsize)

pylab.scatter(x, y, color='gray', s=ssize)
pylab.scatter(x[large], y[large], color='r', s=ssize, label='Above $M_*-R_e$ relation')
pylab.scatter(x[small], y[small], color=colors, marker='o', s=ssize, label='Below $M_*-R_e$ relation')
pylab.axvline(10.9, color='k')
pylab.axvline(11.1, color='k')
pylab.xticks(())
pylab.yticks(())
pylab.xlabel('$\log{M_*}$', fontsize=fsize)
pylab.ylabel('$\log{R_e}$', fontsize=fsize)
pylab.xlim(10.5, 11.5)
pylab.legend(loc = 'upper left', fontsize=16, scatterpoints=1)
pylab.savefig('toy_bias_true.png')
pylab.show()

