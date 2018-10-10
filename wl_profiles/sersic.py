import numpy as np
from scipy.integrate import quad
import os
from scipy.special import gamma as gfunc
from scipy.interpolate import splrep
import ndinterp
import pickle


rgrid_min = 0.001
rgrid_max = 100.
rgrid_n = 1000

ngrid_min = 0.5
ngrid_max = 20.
ngrid_n = 21

def b(nser):
    return 2*nser - 1./3. + 4/405./nser + 46/25515/nser**2

def L(nser, Re):
    return Re**2*2*np.pi*nser/b(nser)**(2*nser)*gfunc(2*nser)

def Sigma(R, nser, Re):
    return np.exp(-b(nser)*(R/Re)**(1./nser))/L(nser, Re)

def M2d(R, nser, Re):
    R = np.atleast_1d(R)
    out = 0.*R
    for i in range(0, len(R)):
        out[i] = 2*np.pi*quad(lambda r: r*Sigma(r, nser, Re), 0., R[i])[0]
    return out

interpname = os.environ.get('BHWLDIR') + '/wl_profiles/sersic_m2d_interp.dat'

if not os.path.isfile(interpname):
    print 'calculating grid of enclosed projected masses...'
    rr = np.logspace(np.log10(rgrid_min), np.log10(rgrid_max), rgrid_n)
    nn = np.logspace(np.log10(ngrid_min), np.log10(ngrid_max), ngrid_n)

    M2d_grid = np.zeros((rgrid_n, ngrid_n))

    for i in range(rgrid_n):
        for j in range(ngrid_n):
            M2d_grid[i, j] = M2d(rr[i], nn[j], 1.)

    axes = {0: splrep(rr, np.arange(rgrid_n)), 1: splrep(nn, np.arange(ngrid_n))}

    M2d_interp = ndinterp.ndInterp(axes, M2d_grid, order=3)
    f = open(interpname, 'w')
    pickle.dump(M2d_interp, f)
    f.close()

    f = open('tmp_grid.dat', 'w')
    pickle.dump(M2d_grid, f)
    f.close()

f = open(interpname, 'r')
M2d_interp = pickle.load(f)
f.close()

def fast_M2d(x, nser):

    xarr = np.atleast_1d(x)
    narr = np.atleast_1d(nser)

    xlen = len(xarr)
    nlen = len(narr)

    if xlen == nlen:
        point = np.array((xarr, narr)).reshape((2, xlen)).T
    elif nlen == 1:
        point = np.array((xarr, nser*np.ones(xlen))).reshape((2, xlen)).T
    elif xlen == 1:
        xarr = x*np.ones(nlen)
        point = np.array((xarr, narr)).reshape((2, nlen)).T
    else:
        print 'unable to match shapes of x and nser'
        df

    oob = xarr > rgrid_max
    oarr = M2d_interp.eval(point)
    oarr[oob] = 1.

    return oarr
