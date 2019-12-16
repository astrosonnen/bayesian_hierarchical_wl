import numpy as np
from scipy.integrate import quad
import os
from scipy.special import gamma as gfunc
from scipy.interpolate import splrep, splev
import pickle


ndeV = 4.

deV_grid_rmin = 0.001
deV_grid_rmax = 1000.
deV_grid_n = 1000

def b(n):
    return 2*n - 1./3. + 4/405./n + 46/25515/n**2

def L(n, Re):
    return Re**2*2*np.pi*n/b(n)**(2*n)*gfunc(2*n)

def Sigma(R, Re):
    return np.exp(-b(ndeV)*(R/Re)**(1./ndeV))/L(ndeV, Re)

def M2d(R, Re):
    R = np.atleast_1d(R)
    out = 0.*R
    for i in range(0, len(R)):
        out[i] = 2*np.pi*quad(lambda r: r*Sigma(r, Re), 0., R[i])[0]
    return out

splinename = os.environ.get('BHWLDIR') + '/wl_profiles/deV_m2d_spline.dat'

if not os.path.isfile(splinename):
    print('calculating grid of enclosed projected masses...')
    rr = np.logspace(np.log10(deV_grid_rmin), np.log10(deV_grid_rmax), deV_grid_n)

    M2d_grid = 0.*rr

    for j in range(deV_grid_n):
        M2d_grid[j] = M2d(rr[j], 1.)

    M2d_spline = splrep(np.array([0.] + list(rr) + [1e10]), np.array([0.] + list(M2d_grid) + [1.]))
    f = open(splinename, 'wb')
    pickle.dump(M2d_spline, f)
    f.close()

f = open(splinename, 'rb')
deV_M2d_spline = pickle.load(f)
f.close()

def fast_M2d(x):
    return splev(x, deV_M2d_spline)

