import numpy as np
from scipy.integrate import quad
import os
from scipy.special import gamma as gfunc
from scipy.interpolate import splrep, splev, splint
import h5py


ndeV = 4.

rgrid_min = 0.001
rgrid_max = 1000.
rgrid_n = 1000

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

gridname = os.environ.get('BHWLDIR') + '/wl_profiles/deV_m2d_grid.hdf5'

if not os.path.isfile(gridname):
    print('calculating grid of enclosed projected masses...')
    rr = np.logspace(np.log10(rgrid_min), np.log10(rgrid_max), rgrid_n)

    M2d_grid = 0.*rr

    for j in range(rgrid_n):
        M2d_grid[j] = M2d(rr[j], 1.)

    grid_file = h5py.File(gridname, 'w')
    grid_file.create_dataset('M2d_grid', data=M2d_grid)
    grid_file.create_dataset('R_grid', data=rr)
    grid_file.close()

else:
    grid_file = h5py.File(gridname, 'r')
    M2d_grid = grid_file['M2d_grid'][()]
    rr = grid_file['R_grid'][()]

deV_M2d_spline = splrep(np.array([0.] + list(rr) + [1e10]), np.array([0.] + list(M2d_grid) + [1.]))

def fast_M2d(x):
    return splev(x, deV_M2d_spline)

def rho(r, reff): # 3D density from spherical deprojection
    rhere = np.atleast_1d(r)
    out = 0.*rhere
    deriv = lambda R: -b(ndeV)/ndeV*(R/(reff))**(1/ndeV)/R*Sigma(R, reff)
    for i in range(len(rhere)):
        out[i] = -1./np.pi*quad(lambda R: deriv(R)/(R**2 - rhere[i]**2)**0.5, rhere[i], np.inf)[0]
    return out

grid3dfilename = os.environ.get('BHWLDIR') + '/wl_profiles/deV_3dgrids.hdf5'

if not os.path.isfile(grid3dfilename):
    print('calculating grid of enclosed 3d masses...')
    M3d_grid = np.zeros(rgrid_n)
    rr_ext = np.append(0., rr)

    rho_extgrid = np.zeros(rgrid_n+1)
    for i in range(rgrid_n):
        rho_extgrid[i+1] = rho(rr[i], 1.)
    mprime_spline = splrep(rr_ext, 4.*np.pi*rho_extgrid*rr_ext**2)
    rho_grid = rho_extgrid[1:]
    for i in range(rgrid_n):
        M3d_grid[i] = splint(0., rr[i], mprime_spline)

    grid_file = h5py.File(grid3dfilename, 'w')
    grid_file.create_dataset('m3d_grid', data=M3d_grid)
    grid_file.create_dataset('rho_grid', data=rho_grid)
    grid_file.create_dataset('r_grid', data=rr)
    grid_file.close()

else:
    grid_file = h5py.File(grid3dfilename, 'r')
    M3d_grid = grid_file['m3d_grid'][()].copy()
    rho_grid = grid_file['rho_grid'][()].copy()
    grid_file.close()

rho_spline = splrep(rr, rho_grid)
M3d_spline = splrep(rr, M3d_grid)

def fast_M3d(x): # 3d mass enclosed within radius x=r/reff, normalized to unit total mass.

    xarr = np.atleast_1d(x)
    oarr = splev(xarr, M3d_spline)

    oob_up = xarr > rgrid_max
    oob_dw = xarr < rgrid_min
    oarr[oob_up] = 1.
    oarr[oob_dw] = 0.

    return oarr

def fast_rho(x, ndeV):

    xarr = np.atleast_1d(x)
    oarr = splev(xarr, rho_spline)

    oob_up = xarr > rgrid_max
    oob_dw = xarr < rgrid_min
    oarr[oob_up] = 0.
    oarr[oob_dw] = 0.

    return oarr

