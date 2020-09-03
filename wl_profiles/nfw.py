# scripts to calculate lensing-related quantities for a spherical NFW profile.
# The profile is normalized so that the mass enclosed within a 3D shell of radius equal to the scale radius r_s is one.

import numpy as np


M3d1 = np.log(2.) - 0.5

def gfunc(x):
    x = np.atleast_1d(x)
    g = x*.0
    arr = x[x<1]
    g[x<1] = np.log(arr/2) + 1/np.sqrt(1 - arr**2)*np.arccosh(1/arr)
    arr = x[x==1]
    g[x==1] = 1 + np.log(0.5)
    arr = x[x>1]
    g[x>1] = np.log(arr/2) + 1/np.sqrt(arr**2-1)*np.arccos(1/arr)
    return g

def Ffunc(x):
    x = np.atleast_1d(x)
    c1 = x<1
    c2 = x==1
    c3 = x>1
    x[c1] = 1/(x[c1]**2-1)*(1 - 1/np.sqrt(1-x[c1]**2)*np.arccosh(1/x[c1]))
    x[c2] = 1/3.
    x[c3] = 1/(x[c3]**2-1)*(1 - 1/np.sqrt(x[c3]**2-1)*np.arccos(1/x[c3]))
    return x

def hfunc(x):
    x = np.atleast_1d(x)
    h = x*.0
    arr = x[x<1]
    h[x<1] = np.log(arr/2)**2 - np.arccosh(1/arr)**2
    arr = x[x>=1]
    h[x>=1] = np.log(arr/2)**2 - np.arccos(1/arr)**2
    return h


def rho(r,rs):
    return 1./(r/rs)/(1 + r/rs)**2/(4*np.pi*rs**3)/M3d1

def M3d(r,rs):
    return (np.log(1 + r/rs) - r/(r+rs))/M3d1

def Sigma(r,rs):
    return Ffunc(r/rs)/(2*np.pi*rs**2)/M3d1

def M2d(r,rs):
    return gfunc(r/rs)/M3d1

def lenspot(r,rs):
    return hfunc(r/rs)/(2.*np.pi)/M3d1


