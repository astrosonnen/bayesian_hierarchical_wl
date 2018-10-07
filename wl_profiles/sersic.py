import numpy as np
from scipy.integrate import quad
import os
from scipy.special import gamma as gfunc


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


