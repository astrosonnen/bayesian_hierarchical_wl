import numpy as np
import h5py
import os
import sys
import wl_lens_models


mockname = 'isolated_gnfw_fixedz_mock'

batchno = int(sys.argv[1])
batchsize = 100
# reads in mock lens catalog
mock = h5py.File('../wl_sims/%s.hdf5'%mockname, 'r')

griddir = '/net/ringvaart/data2/sonnenfeld/4hs_forecasts/%s_grids/'%mockname # directory where grids will be stored

if not os.path.isdir(griddir):
    os.system('mkdir %s'%griddir)

zd = mock.attrs['zd']
zs = mock.attrs['zs']
c200 = mock.attrs['c200']
gammadm_min = mock.attrs['gammadm_min']
gammadm_max = mock.attrs['gammadm_max']
sigma_eps = mock.attrs['sigma_eps']

nlens = mock.attrs['nlens']

istart = batchno * batchsize
iend = min(nlens, (batchno+1)*batchsize)

# defines grid axes
m200_min = 11.
m200_max = 15.
nm200 = 41

m200_grid = np.linspace(m200_min, m200_max, nm200)

mstar_min = 10.5
mstar_max = 13.
nmstar = 26

mstar_grid = np.linspace(mstar_min, mstar_max, nmstar)

ngammadm = 21

gammadm_grid = np.linspace(gammadm_min, gammadm_max, ngammadm)

lens_model = wl_lens_models.GNFWdeV(z=zd, c200=c200)
s_cr = lens_model.S_cr(zs) # critical surface mass density for lensing (M_Sun/Mpc^2)

for i in range(istart, iend):

    print(i)
    lensname = 'lens_%05d'%i
    gridname = griddir+'%s_wl_likegrid.hdf5'%lensname

    lens_model.reff = 10.**mock['lreff_samp'][i]

    group = mock[lensname]
    theta_source = group['R_source'][()]
    et_obs = group['et_obs'][()]
    
    like_grid = np.zeros((nmstar, nm200, ngammadm))
    
    for j in range(nmstar):
        lens_model.mstar = 10.**mstar_grid[j]
        for k in range(nm200):
            lens_model.m200 = 10.**m200_grid[k]
            for l in range(ngammadm):
                lens_model.gammadm = gammadm_grid[l]
                lens_model.update()
   
                gammat = lens_model.gammat(theta_source, s_cr)
                kappa = lens_model.kappa(theta_source, s_cr)

                et_model = gammat/(1.-kappa)
    
                like_grid[j, k, l] = (-0.5*(et_model - et_obs)**2 / sigma_eps**2).sum()

    grid_file = h5py.File(gridname, 'w')
    
    grid_file.create_dataset('mstar_grid', data=mstar_grid)
    grid_file.create_dataset('m200_grid', data=m200_grid)
    grid_file.create_dataset('gammadm_grid', data=gammadm_grid)
    grid_file.create_dataset('wl_like_grid', data=like_grid)

    grid_file.close()

