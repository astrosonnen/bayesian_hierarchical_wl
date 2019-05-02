import numpy as np
import h5py
import os
import sys
import wl_lens_models


griddir = 'nfw_grids/'

# Grid boundaries
m200_min = 11.
m200_max = 16.
nm200 = 51

m200_grid = np.linspace(m200_min, m200_max, nm200)

mstar_min = 10.5
mstar_max = 13.
nmstar = 26

mstar_grid = np.linspace(mstar_min, mstar_max, nmstar)

lc200_min = 0.
lc200_max = 2.
nc200 = 11

lc200_grid = np.linspace(lc200_min, lc200_max, nc200)

# READ IN LENS CATALOG
f = open('sdss_legacy_hscoverlap_mcut11.0.cat', 'r')
zd, mchab, mchab_err = np.loadtxt(f, usecols=(2, 3, 4), unpack=True)
f.close()

ngal = len(zd)

# READ IN SOURCE CATALOG
source_file = h5py.File('sources.hdf5', 'r')

for i in range(ngal):

    print i
    lensname = 'lens_%04d'%i
    gridname = griddir+'%s_nfw_grid.hdf5'%lensname

    if lensname in source_file:
        if len(source_file[names[i]]['r']) > 0: # checks if there are any sources behind this lens

            model_lens = wl_lens_models.NFWPoint(z=zd[i], m200=1., c200=1., mstar=1.)
            group = source_file[names[i]]
    
            sources = {}
            sources['r'] = group['r'].value.copy()
            sources['et'] = group['et'].value.copy()
            sources['et_err'] = (group['rms_e'].value**2 + group['sigma_e'].value**2)**0.5
            sources['R'] = 1. - group['rms_e'].value**2
            sources['ct_bias'] = group['ct_bias'].value.copy()
            sources['m_bias'] = group['m_bias'].value.copy()
            sources['s_cr'] = group['s_cr'].value.copy()
    
            model_lens.sources = sources

            like_grid = np.zeros((nmstar, nm200, nc200))
    
            for j in range(nmstar):
                model_lens.mstar = 10.**mstar_grid[j]
                for k in range(nm200):
                    model_lens.m200 = 10.**m200_grid[k]
                    for l in range(nc200):
                        model_lens.c200 = 10.**lc200_grid[l]
                        model_lens.update()
   
                        model_lens.get_source_kappa()
                        model_lens.get_source_gammat()
            
                        gt_model = model_lens.sources['gammat']/(1. - model_lens.sources['kappa'])
                        et_model = 2.*model_lens.sources['R'] * ((1. + model_lens.sources['m_bias']) * gt_model + model_lens.sources['ct_bias'])
            
                        like_grid[j, k, l] = (-0.5*(et_model - model_lens.sources['et'])**2 / model_lens.sources['et_err']**2).sum()

            grid_file = h5py.File(gridname, 'w')
    
            grid_file.create_dataset('mstar_grid', data=mstar_grid)
            grid_file.create_dataset('m200_grid', data=m200_grid)
            grid_file.create_dataset('lc200_grid', data=lc200_grid)
            grid_file.create_dataset('wl_like_grid', data=like_grid)
    
            grid_file.close()

