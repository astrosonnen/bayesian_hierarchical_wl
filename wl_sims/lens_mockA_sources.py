# these functions generate mock data
from config import *
import numpy as np
import lens_tools
import h5py
from cosmolopy import distance


mockname = 'mockAB'

cosmo = {'omega_M_0': omegaM, 'omega_lambda_0': omegaL, 'omega_k_0': omegak, 'h': h}

pop_file = h5py.File('mockAB_pop.hdf5', 'r')

source_file = h5py.File('mockAB_sources_dsspline.hdf5', 'r')

lensed_source_file = h5py.File('mockA_lensed_sources_dsspline.hdf5', 'w')

for par in ['zs_err', 'rmax', 'nbkg', 'sigma_eps']:
    lensed_source_file.create_dataset(par, data=source_file[par].value)

nlens = len(pop_file['mhalo'])

for i in range(nlens):

    print i

    lensname = 'lens_%05d'%i
    lensed_source_file.create_group(lensname)

    lens = lens_tools.NFWdeVProfile(z=pop_file['zd'][i], m200=10.**pop_file['mhalo'][i], \
                                    c200=10.**pop_file['c200'][i], mstar=10.**pop_file['mstar'][i], reff=pop_file['reff'][i], ra=pop_file['ra'][i], \
                                    dec=pop_file['dec'][i])

    bkg = source_file[lensname]['z'].value > pop_file['zd'][i] + 0.01

    nsource = len(source_file[lensname]['z'])
    nbkg = len(source_file[lensname]['z'][bkg])

    kappa_bkg = np.zeros(nbkg)
    gamma_bkg = 0.*kappa_bkg + 1j*0.*kappa_bkg

    compell = source_file[lensname]['e1s'].value.copy() + 1j*source_file[lensname]['e2s'].value.copy()

    bkg_sources = {}

    for par in ['s_cr', 'ra', 'dec']:
        bkg_sources[par] = source_file[lensname][par].value[bkg]

    lens.sources = bkg_sources

    lens.get_source_polarcoords()

    kappa_bkg += lens.sources_kappa()

    gamma_bkg += lens.get_source_gamma()

    gcompl_bkg = gamma_bkg / (1. - kappa_bkg)

    compell[bkg] = (compell[bkg] + gcompl_bkg) / (1. + np.conj(compell[bkg])*np.conj(gcompl_bkg))

    e1 = np.real(compell)
    e2 = np.imag(compell)

    sin2phi = 2.*np.cos(source_file[lensname]['phi'])*np.sin(source_file[lensname]['phi'])
    cos2phi = np.cos(source_file[lensname]['phi'])**2 - np.sin(source_file[lensname]['phi'])**2

    et = e1*cos2phi + e2*sin2phi
    er = -e1*sin2phi + e2*cos2phi

    for par in source_file[lensname]:
        lensed_source_file[lensname].create_dataset(par, data=source_file[lensname][par].value)

    lensed_source_file[lensname].create_dataset('e1', data=e1)
    lensed_source_file[lensname].create_dataset('e2', data=e2)
    lensed_source_file[lensname].create_dataset('et', data=et)
    lensed_source_file[lensname].create_dataset('er', data=er)

lensed_source_file.close()
