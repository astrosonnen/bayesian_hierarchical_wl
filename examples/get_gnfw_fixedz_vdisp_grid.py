import numpy as np
import h5py
import os
import sys
from wl_profiles import sersic, gnfw
import wl_cosmology
from wl_cosmology import Mpc, M_Sun, G
import sigma_model
from tracer_profiles import deVaucouleurs


mockname = 'isolated_gnfw_fixedz_wdyn'

# reads in mock lens catalog
mock = h5py.File('../wl_sims/%s_mock.hdf5'%mockname, 'r')

outdir = '/net/ringvaart/data2/sonnenfeld/4hs_forecasts/' # directory where grid will be stored
outname = outdir+'/%s_dyngrid.hdf5'%mockname

zd = mock.attrs['zd']
c200 = mock.attrs['c200']
gammadm_min = mock.attrs['gammadm_min']
gammadm_max = mock.attrs['gammadm_max']
nser = mock.attrs['nser']

dd = wl_cosmology.Dang(zd)
rhoc = wl_cosmology.rhoc(zd) # critical density of the Universe at z=zd. Halo masses are defined as M200 wrt rhoc.

kpc = Mpc/1000.
arcsec2rad = np.deg2rad(1./3600.)
arcsec2kpc = arcsec2rad * dd * 1000.

fiber_radius_kpc = 1. * arcsec2kpc
nr = 1000 # number of points in radial grid for Jeans code

# velocity dispersion of stellar component (for unit mass, on a grid of half-light radius)
lreff_min = 0.99 * mock['lreff_samp'][()].min()
lreff_max = 1.01 * mock['lreff_samp'][()].max()

nreff = 21
lreff_grid = np.linspace(lreff_min, lreff_max, nreff)

s2_deV_grid = np.zeros(nreff)

rgrid_scalefree = np.logspace(np.log10(sersic.rgrid_min), np.log10(sersic.rgrid_max), nr)
m3d_deV_scalefree = sersic.fast_M3d(rgrid_scalefree, nser)

print('calculating velocity dispersion of stellar component...')
for i in range(nreff):
    rgrid_here = 10.**lreff_grid[i] * rgrid_scalefree
    s2_deV = sigma_model.sigma2((rgrid_here, m3d_deV_scalefree), fiber_radius_kpc, 10.**lreff_grid[i], deVaucouleurs)
    s2_deV_grid[i] = s2_deV*G*M_Sun/kpc / 1e10

# defines grid axes
lm200_min = 11.
lm200_max = 15.
nm200 = 21

lm200_grid = np.linspace(lm200_min, lm200_max, nm200)

ngammadm = 11

gammadm_grid = np.linspace(gammadm_min, gammadm_max, ngammadm)

s2_gnfw_grid = np.zeros((nm200, ngammadm, nreff))

rgrid_gnfw = np.logspace(np.log10(fiber_radius_kpc) - 2., np.log10(fiber_radius_kpc)+2., nr)

print('calculating velocity dispersion of halo...')
for i in range(nm200):
    print(i)
    r200 = (10.**lm200_grid[i]*3./200./(4.*np.pi)/rhoc)**(1./3.) * 1000. # virial radius in kpc
    rs = r200/c200 # halo scale radius in kpc

    Rmin_here = rs * gnfw.Rgrid_min
    Rmax_here = rs * gnfw.Rgrid_max

    good = (rgrid_gnfw > Rmin_here) & (rgrid_gnfw < Rmax_here)

    rgrid_gnfw_here = rgrid_gnfw[good]

    for j in range(ngammadm):
        gnfw_norm = 10.**lm200_grid[i] / gnfw.M3d(r200, rs, gammadm_grid[j])

        m3d_gnfw_here = gnfw_norm * gnfw.fast_M3d(rgrid_gnfw_here, rs, gammadm_grid[j])

        for k in range(nreff):
            s2_here = sigma_model.sigma2((rgrid_gnfw_here, m3d_gnfw_here), fiber_radius_kpc, 10.**lreff_grid[k], deVaucouleurs)
            s2_gnfw_grid[i, j, k] = s2_here*G*M_Sun/kpc / 1e10

grid_file = h5py.File(outname, 'w')
    
grid_file.create_dataset('lreff_grid', data=lreff_grid)
grid_file.create_dataset('s2_deV_grid', data=s2_deV_grid)

grid_file.create_dataset('lm200_grid', data=lm200_grid)
grid_file.create_dataset('gammadm_grid', data=gammadm_grid)
grid_file.create_dataset('s2_gnfw_grid', data=s2_gnfw_grid)
grid_file.close()

