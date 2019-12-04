import numpy as np
from scipy.integrate import quad
import pickle
import os
from scipy.interpolate import splrep

#calculates density profiles, projected mass densities, projected enclosed masses, 3d enclosed masses for generalized-NFW profiles.

grid_dir = os.environ.get('BHWLDIR') + '/wl_profiles/'

def rho(r, rs, beta):
    return 1./r**beta/(1. + r/rs)**(3.-beta) * rs**(beta-3.)

def Sigma(R, rs, beta):
    Rs = np.atleast_1d(R)
    out = 0.*Rs
    norm = 0.5*rs**2
    for i in range(0,len(Rs)):
        R = Rs[i]
        out[i] = (R/rs)**(1-beta)*quad(lambda theta: np.sin(theta)*(np.sin(theta) + R/rs)**(beta-3),0.,np.pi/2.)[0]
    return out/norm

def M2d(R, rs, beta):
    Rs = np.atleast_1d(R)
    out = 0.*Rs
    for i in range(0,len(Rs)):
        R = Rs[i]
        out[i] = 2*np.pi*quad(lambda x: Sigma(x, rs, beta)*x, 0., R)[0]
    return out

def M3d(r, rs, beta):
    r = np.atleast_1d(r)
    out = 0.*r
    for i in range(0, len(r)):
        out[i] = 4*np.pi*quad(lambda x: rho(x, rs, beta)*x**2, 0., r[i])[0]
    return out

def fast_M2d(R, rs, beta):
    R = np.atleast_1d(R)
    beta = np.atleast_1d(beta)
    length = max(len(beta), len(R))
    sample = np.array([beta*np.ones(length), R/rs*np.ones(length)]).reshape((2, length)).T
    M2d = M2d_grid.eval(sample)*(R/rs)**(3.-beta)
    return M2d

def fast_M3d(r, rs, beta):
    r = np.atleast_1d(r)
    beta = np.atleast_1d(beta)
    length = max(len(beta), len(r))
    sample = np.array([beta*np.ones(length), r/rs*np.ones(length)]).reshape((2, length)).T
    M3d = M3d_grid.eval(sample)*(r/rs)**(3.-beta)
    return M3d

def fast_Sigma(R, rs, beta):
    R = np.atleast_1d(R)
    beta = np.atleast_1d(beta)
    length = max(len(beta), len(R))
    sample = np.array([beta*np.ones(length), R/rs*np.ones(length)]).reshape((2, length)).T
    Sigma = Sigma_grid.eval(sample)/(R/rs)/rs**2
    return Sigma

def make_M2d_Rbetam3_grid(Nr=100, Nb=28, Rmin=0.001, Rmax=100.):
    #this code calculates the quantity M2d(R, rs=1, beta)*R**(3-beta) on a grid of values of R between Rmin and Rmax, and values of the inner slope beta between 0.1 and 2.8.
    #the reason for the multiplication by R**(3-beta) is to make interpolation easier by having a function as flat as possible.

    print('calculating grid of enclosed projected masses...')
    import ndinterp
    R_grid = np.logspace(np.log10(Rmin), np.log10(Rmax), Nr)
    R_spline = splrep(R_grid, np.arange(Nr))
    beta_grid = np.linspace(0.1, 2.8, Nb)
    beta_spline = splrep(beta_grid, np.arange(Nb))
    axes = {0:beta_spline, 1:R_spline}

    R,B = np.meshgrid(R_grid, beta_grid)
    M2d_grid = np.empty((Nb, Nr))
    
    for i in range(Nb):
        print('inner slope %4.2f'%beta_grid[i])
        for j in range(Nr):
            M2d_grid[i,j] = M2d(R_grid[j], 1., beta_grid[i])
    thing = ndinterp.ndInterp(axes, M2d_grid*R**(B-3.), order=3)
    f = open(grid_dir+'/gNFW_M2d_Rbetam3_grid.dat','wb')
    pickle.dump(thing, f)
    f.close()

def make_M3d_rbetam3_grid(Nr=100, Nb=28, rmin=0.01, rmax=100.):
    import ndinterp
    #this code calculates the quantity M3d(R, rs=1, beta)*R**(3-beta) on a grid of values of R between Rmin and Rmax, and values of the inner slope beta between 0.1 and 2.8.
    #the reason for the multiplication by R**(3-beta) is to make interpolation easier by having a function as flat as possible.

    print('calculating grid of enclosed projected masses...')
    import ndinterp
    r_grid = np.logspace(np.log10(rmin), np.log10(rmax), Nr)
    r_spline = splrep(r_grid, np.arange(Nr))
    beta_grid = np.linspace(0.1, 2.8, Nb)
    beta_spline = splrep(beta_grid, np.arange(Nb))
    axes = {0:beta_spline, 1:r_spline}

    R,B = np.meshgrid(r_grid, beta_grid)
    M3d_grid = np.empty((Nb, Nr))

    for i in range(Nb):
        print('inner slope %4.2f'%beta_grid[i])
        for j in range(Nr):
            M3d_grid[i,j] = M3d(r_grid[j], 1., beta_grid[i])
    thing = ndinterp.ndInterp(axes, M3d_grid*R**(B-3.), order=3)
    f = open(grid_dir+'/gNFW_M3d_rbetam3_grid.dat','wb')
    pickle.dump(thing, f)
    f.close()

def make_Sigma_R_grid(Nr=100, Nb=28, Rmin=0.01, Rmax=100.):
    #this code calculates the quantity M2d(R, rs=1, beta)*R**(3-beta) on a grid of values of R between Rmin and Rmax, and values of the inner slope beta between 0.1 and 2.8.
    #the reason for the multiplication by R**(3-beta) is to make interpolation easier by having a function as flat as possible.

    print('calculating grid of projected masses...')
    import ndinterp
    R_grid = np.logspace(np.log10(Rmin), np.log10(Rmax), Nr)
    R_spline = splrep(R_grid, np.arange(Nr))
    beta_grid = np.linspace(0.1, 2.8, Nb)
    beta_spline = splrep(beta_grid, np.arange(Nb))
    axes = {0:beta_spline, 1:R_spline}

    R,B = np.meshgrid(R_grid, beta_grid)
    Sigma_grid = np.empty((Nb, Nr))
    
    for i in range(Nb):
        print('inner slope %4.2f'%beta_grid[i])
        for j in range(Nr):
            Sigma_grid[i,j] = Sigma(R_grid[j], 1., beta_grid[i])
    thing = ndinterp.ndInterp(axes, Sigma_grid*R, order=3)
    f = open(grid_dir+'/gNFW_Sigma_R_grid.dat','wb')
    pickle.dump(thing, f)
    f.close()

if not os.path.isfile(grid_dir+'/gNFW_M2d_Rbetam3_grid.dat'):
    make_M2d_Rbetam3_grid()

if not os.path.isfile(grid_dir+'/gNFW_M3d_rbetam3_grid.dat'):
    make_M3d_rbetam3_grid()

if not os.path.isfile(grid_dir+'/gNFW_Sigma_R_grid.dat'):
    make_Sigma_R_grid()

f = open(grid_dir+'/gNFW_M2d_Rbetam3_grid.dat','rb')
M2d_grid = pickle.load(f)
f.close()

f = open(grid_dir+'/gNFW_M3d_rbetam3_grid.dat','rb')
M3d_grid = pickle.load(f)
f.close()

f = open(grid_dir+'/gNFW_Sigma_R_grid.dat','rb')
Sigma_grid = pickle.load(f)
f.close()

