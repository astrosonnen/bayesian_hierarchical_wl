import numpy as np
import h5py
import pymc
import os
import sys
import wl_lens_models


chaindir = 'nfw_chains/'

# MCMC parameters
nsamp = 11000
burnin = 1000
thin = 10 # THINNING: KEEP ONLY 1 IN 10 STEPS IN MCMC CHAIN. 1000 POINT IS SUFFICIENT FOR MC INTEGRAL

# READ IN LENS CATALOG
f = open('sdss_legacy_hscoverlap_mcut11.0.cat', 'r')
zd, mchab, mchab_err = np.loadtxt(f, usecols=(2, 3, 4), unpack=True)
f.close()

ngal = len(zd)

# READ IN SOURCE CATALOG
source_file = h5py.File('camira_specz_sources.hdf5', 'r')

# interim priors on m200 and c200: these should be sufficiently broad to cover the area of parameter space spanned by the true values...
m200_prior = {'type': 'Gauss', 'mu': 13.5, 'sigma': 1.}
c200_prior = {'type': 'Gauss', 'mu': 0.6, 'sigma': 0.3}

for i in range(ngal):

    print i
    name = 'lens_%04d'%i
    chainname = chaindir+'%s_nfw_chain.hdf5'%name
    
    if name in source_file:
        if len(source_file[name]['r']) > 0: # checks if there are any sources behind this lens
            chain_file = h5py.File(chainname, 'w')
    
            model_lens = wl_lens_models.NFWPoint(z=zd[i], m200=1., c200=1., mstar=1.)
            group = source_file[name]
    
            sources = {}
            sources['r'] = group['r'].value.copy()
            sources['et'] = group['et'].value.copy()
            sources['et_err'] = (group['rms_e'].value**2 + group['sigma_e'].value**2)**0.5
            sources['R'] = 1. - group['rms_e'].value**2
            sources['ct_bias'] = group['ct_bias'].value.copy()
            sources['m_bias'] = group['m_bias'].value.copy()
            sources['s_cr'] = group['s_cr'].value.copy()
    
            model_lens.sources = sources
    
            mchab_var = pymc.Uniform('mchab', lower=10., upper=13., value=mchab[i])
    
            m200_var = pymc.Normal('m200', mu=m200_prior['mu'], tau=1./m200_prior['sigma']**2, value=m200_prior['mu'])
            c200_var = pymc.Normal('c200', mu=c200_prior['mu'], tau=1./c200_prior['sigma']**2, value=c200_prior['mu'])
    
            pars = [mchab_var, m200_var, c200_var]
    
            @pymc.deterministic()
            def like(ms=mchab_var, m200=m200_var, c200=c200_var):
        
                model_lens.m200 = 10.**m200
                model_lens.c200 = 10.**c200
                model_lens.mstar = 10.**ms
        
                model_lens.update()
    
                model_lens.get_source_kappa()
                model_lens.get_source_gammat()
    
                gt_model = model_lens.sources['gammat']/(1. - model_lens.sources['kappa'])
                et_model = 2.*model_lens.sources['R'] * ((1. + model_lens.sources['m_bias']) * gt_model + model_lens.sources['ct_bias'])
    
                wl_logp = (-0.5*(et_model - model_lens.sources['et'])**2 / model_lens.sources['et_err']**2).sum()
    
                mchab_logp = -0.5*(ms - mchab[i])**2/mchab_err[i]**2
        
                return wl_logp + mchab_logp
        
            @pymc.stochastic()
            def logp(observed=True, value=0., p=pars):
                return like
        
            M = pymc.MCMC(pars)
            M.use_step_method(pymc.AdaptiveMetropolis, [mchab_var, m200_var, c200_var])
            M.sample(nsamp, burnin, thin=thin)
    
            mp_group = chain_file.create_group('m200_prior')
            for par in m200_prior:
                mp_group.create_dataset(par, data=m200_prior[par])
    
            cp_group = chain_file.create_group('c200_prior')
            for par in c200_prior:
                cp_group.create_dataset(par, data=c200_prior[par])
    
            chain_file.create_dataset('mchab', data=M.trace('mchab')[:])
            chain_file.create_dataset('m200', data=M.trace('m200')[:])
            chain_file.create_dataset('c200', data=M.trace('c200')[:])
    
            chain_file.close()

