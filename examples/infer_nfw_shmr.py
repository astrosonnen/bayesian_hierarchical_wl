import numpy as np
import h5py
import emcee
import os
import pickle
from scipy.stats import truncnorm
from scipy.special import erf


chaindir = '/Users/sonnen/hsc_weaklensing/sdss_legacy/nfw_chains/' # <-- CHANGE PATH

# reads in lens catalog
f = open('sdss_legacy_hscoverlap_mcut11.0.cat', 'r')
zd, mchab, merr = np.loadtxt(f, usecols=(2, 3, 4), unpack=True)
f.close()

ngal = len(names)

chains = []
mchab_samp = []
merr_samp = []

mchab_cut = 11.
mchab_piv = 11.3

# reads individual lens mcmc chains
for i in range(ngal):
    name = 'lens_%04d'%i
    print i
    chainname = chaindir+'%s_vanilla_chab_broadcprior.hdf5'%name
        
    if os.path.isfile(chainname):
        chain_file = h5py.File(chainname, 'r')

        chain = {}
        chain['m200'] = chain_file['m200'].value.copy()
        chain['c200'] = chain_file['c200'].value.copy()
        chain['mchab'] = chain_file['mchab'].value.copy()

        chain['m200_prior'] = (chain_file['m200_prior']['mu'].value, chain_file['m200_prior']['sigma'].value)
        chain['c200_prior'] = (chain_file['c200_prior']['mu'].value, chain_file['c200_prior']['beta'].value, chain_file['c200_prior']['pivot'].value, chain_file['c200_prior']['sigma'].value)

        chains.append(chain)
        chain_file.close()

        mchab_samp.append(mchab[i])
        merr_samp.append(merr[i])

mchab_samp = np.array(mchab_samp)
merr_samp = np.array(merr_samp)

nstep = 500 # NUMBER OF STEPS IN MCMC

# DEFINE MODEL HYPER-PARAMETERS
m200_mu = {'name': 'mu', 'lower': 11., 'upper': 15., 'guess': 13.7, 'step': 0.1}
m200_sig = {'name': 'sig', 'lower': 0., 'upper': 1., 'guess': 0.4, 'step': 0.03}
mchab_dep = {'name': 'mchab_dep', 'lower': 0., 'upper': 5., 'guess': 2., 'step': 0.3}

mchab_mu = {'name': 'mchab_mu', 'lower': 10., 'upper': 12., 'guess': mchab_samp.mean(), 'step': 0.03}
mchab_sig = {'name': 'mchab_sig', 'lower': 0., 'upper': 1., 'guess': mchab_samp.std(), 'step': 0.03}
mchab_logskew = {'name': 'mchab_logskew', 'lower': -1., 'upper': 1., 'guess': 0., 'step': 0.03}

c200_mu = {'name': 'c200_mu', 'lower': 0., 'upper': 2., 'guess': 0.7, 'step': 0.1}
c200_sig = {'name': 'c200_sig', 'lower': 0., 'upper': 1., 'guess': 0.1, 'step': 0.03}

pars = [m200_mu, m200_sig, mchab_dep, mchab_mu, mchab_sig, mchab_logskew, c200_mu, c200_sig]

npars = len(pars)

nwalkers = 6*npars # NUMBER OF WALKERS

bounds = []
for par in pars:
    bounds.append((par['lower'], par['upper']))

nlens = len(chains)

def logprior(p):
    for i in range(npars):
        if p[i] < bounds[i][0] or p[i] > bounds[i][1]:
            return -1e300
    return 0.

def logpfunc(p):

    lprior = logprior(p)
    if lprior < 0.:
        return -1e300

    mu, sig, mchab_dep, mchab_mu, mchab_sig, mchab_logskew, c200_mu, c200_sig = p

    logp = 0.
    for i in range(nlens):

        m200_muhere = mu + mchab_dep*(chains[i]['mchab'] - mchab_piv)
        mchab_term = 1./(2.*np.pi)**0.5/mchab_sig * np.exp(-0.5*(chains[i]['mchab'] - mchab_mu)**2/mchab_sig**2) * 0.5 * (1. + erf((chains[i]['mchab'] - mchab_mu)/mchab_sig/2.**0.5*10.**mchab_logskew))

        m200_term = 1./(2.*np.pi)**0.5/sig * np.exp(-0.5*(chains[i]['m200'] - m200_muhere)**2/sig**2)

        c200_term = 1./(2.*np.pi)**0.5/c200_sig * np.exp(-0.5*(chains[i]['c200'] - c200_mu)**2/c200_sig**2)

        cut_renorm = 0.5*(erf((chains[i]['mchab'] - mchab_cut)/2.**0.5/merr_samp[i]) + 1.) # normalizes likelihood to 1, taking into account the cut in observed stellar mass

        interim_prior = 1./(2.*np.pi)/chains[i]['m200_prior'][1]/chains[i]['c200_prior'][1]*\
                        np.exp(-0.5*(chains[i]['m200_prior'][0] - chains[i]['m200'])**2/chains[i]['m200_prior'][1]**2) * \
                        np.exp(-0.5*(chains[i]['c200_prior'][0] - chains[i]['c200'] )**2/chains[i]['c200_prior'][1]**2)

        integrand = mchab_term * m200_term * c200_term / interim_prior / cut_renorm

        logp += np.log(integrand.sum())

    if logp != logp:
        return -1e300

    return logp

sampler = emcee.EnsembleSampler(nwalkers, npars, logpfunc)

start = []
for i in range(nwalkers):
    tmp = np.zeros(npars)
    for j in range(npars):
        a, b = (bounds[j][0] - pars[j]['guess'])/pars[j]['step'], (bounds[j][1] - pars[j]['guess'])/pars[j]['step']
        p0 = truncnorm.rvs(a, b, size=1)*pars[j]['step'] + pars[j]['guess']
        tmp[j] = p0

    start.append(tmp)

print "Sampling"

sampler.run_mcmc(start, nstep)

output = {}
output['logp'] = sampler.lnprobability
for n in range(npars):
    output[pars[n]['name']] = sampler.chain[:, :, n]

f = open('nfw_shmr_inference.dat', 'w')
pickle.dump(output, f)
f.close()

