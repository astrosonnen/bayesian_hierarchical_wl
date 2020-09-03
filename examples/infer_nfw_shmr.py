import numpy as np
import h5py
import emcee
import os
import pickle
from scipy.stats import truncnorm
from scipy.interpolate import splrep
from scipy.special import erf
import ndinterp


griddir = 'nfw_grids/' # <-- CHANGE PATH

# reads in lens catalog
f = open('sdss_legacy_hscoverlap_mcut11.0.cat', 'r')
zd, mstar, merr = np.loadtxt(f, usecols=(2, 3, 4), unpack=True)
f.close()

ngal = len(zd)

nint = 1000

mstar_cut = 11.
mstar_piv = 11.3

grids = []

mstar_impsamp = []
m200_impsamp = []
lc200_impsamp = []

mstar_samp = []
merr_samp = []

mstar_grids = []
m200_grids = []
lc200_grids = []

# reads individual lens likelihood grids
for i in range(ngal):
    name = 'lens_%04d'%i
    print i
    gridname = griddir+'/%s_nfw_grid.hdf5'%name

    if os.path.isfile(gridname):
        grid_file = h5py.File(gridname, 'r')

        # READS IN GRID AXES
        mstar_grid = grid_file['mstar_grid'][()]
        mstar_grids.append(mstar_grid)

        m200_grid = grid_file['m200_grid'][()]
        m200_grids.append(m200_grid)

        lc200_grid = grid_file['lc200_grid'][()]
        lc200_grids.append(lc200_grid)

        axes = {0: splrep(mstar_grid, np.arange(len(mstar_grid))), 1: splrep(m200_grid, np.arange(len(m200_grid))), 2: splrep(lc200_grid, np.arange(len(lc200_grid)))}

        grid_here = grid_file['wl_like_grid'][()]
        grid_here -= grid_here.max()

        # PREPARES A 3D INTERPOLATOR OBJECT ON THE GRID
        grid = ndinterp.ndInterp(axes, grid_here, order=1)
        grids.append(grid)

        # PREPARES SCALE-FREE GAUSSIAN SAMPLES FOR MC INTEGRATION
        mstar_impsamp.append(np.random.normal(0., 1., nint))
        m200_impsamp.append(np.random.normal(0., 1., nint))
        lc200_impsamp.append(np.random.normal(0., 1., nint))

        grid_file.close()

        mstar_samp.append(mstar[i])
        merr_samp.append(merr[i])

mstar_samp = np.array(mstar_samp)
merr_samp = np.array(merr_samp)

nstep = 500 # NUMBER OF STEPS IN MCMC

# DEFINE MODEL HYPER-PARAMETERS
m200_mu = {'name': 'm200_mu', 'lower': 11., 'upper': 15., 'guess': 13.7, 'step': 0.1}
m200_sig = {'name': 'm200_sig', 'lower': 0., 'upper': 1., 'guess': 0.4, 'step': 0.03}
m200_mstar_dep = {'name': 'm200_mstar_dep', 'lower': 0., 'upper': 5., 'guess': 2., 'step': 0.3}

mstar_mu = {'name': 'mstar_mu', 'lower': 10., 'upper': 12., 'guess': mstar_samp.mean(), 'step': 0.03}
mstar_sig = {'name': 'mstar_sig', 'lower': 0., 'upper': 1., 'guess': mstar_samp.std(), 'step': 0.03}

c200_mu = {'name': 'c200_mu', 'lower': 0., 'upper': 2., 'guess': 0.7, 'step': 0.1}
c200_sig = {'name': 'c200_sig', 'lower': 0., 'upper': 1., 'guess': 0.1, 'step': 0.03}

pars = [m200_mu, m200_sig, m200_mstar_dep, mstar_mu, mstar_sig, c200_mu, c200_sig]

npars = len(pars)

nwalkers = 6*npars # NUMBER OF WALKERS

bounds = []
for par in pars:
    bounds.append((par['lower'], par['upper']))

nlens = len(grids)

def logprior(p):
    for i in range(npars):
        if p[i] < bounds[i][0] or p[i] > bounds[i][1]:
            return -1e300
    return 0.

def logpfunc(p):

    lprior = logprior(p)
    if lprior < 0.:
        return -1e300

    m200_mu, m200_sig, m200_mstar_dep, mstar_mu, mstar_sig, c200_mu, c200_sig = p

    logp = 0.
    for i in range(nlens):

        # RESCALES THE SCALE-FREE SAMPLES TO THE MODEL DISTRIBUTION
        # (distribution is artificially truncated at the grid boundaries)
        mstar_here = mstar_mu + mstar_impsamp[i] * mstar_sig
        mstar_here[mstar_here < mstar_grids[i][0]] = mstar_grids[i][0]
        mstar_here[mstar_here > mstar_grids[i][-1]] = mstar_grids[i][-1]

        m200_muhere = m200_mu + m200_mstar_dep*(mstar_here - mstar_piv)
        m200_here = m200_muhere + m200_sig * m200_impsamp[i]
        m200_here[m200_here < m200_grids[i][0]] = m200_grids[i][0]
        m200_here[m200_here > m200_grids[i][-1]] = m200_grids[i][-1]

        lc200_here = c200_mu + c200_sig * lc200_impsamp[i]
        lc200_here[lc200_here > lc200_grids[i][-1]] = lc200_grids[i][-1]
        lc200_oob = lc200_here < lc200_grids[i][0]
        lc200_here[lc200_oob] = lc200_grids[i][0]
        
        # OBSERVED STELLAR MASS LIKELIHOOD TERM
        mstar_like = 1./(2.*np.pi)**0.5/merr_samp[i] * np.exp(-0.5*(mstar_here - mstar_samp[i])**2/merr_samp[i]**2)

        # DEFINES ARRAY FOR EVALUATION OF WEAK LENSING LIKELIHOOD
        point = np.array((mstar_here, m200_here, lc200_here)).T

        # EVALUATES WEAK LENSING LIKELIHOOD ON GRID
        wl_loglike = grids[i].eval(point)

        cut_renorm = 0.5*(erf((mstar_impsamp[i] - mstar_cut)/2.**0.5/merr_samp[i]) + 1.) # normalizes likelihood to 1, taking into account the cut in observed stellar mass

        integrand = mstar_like * np.exp(wl_loglike) / cut_renorm
        integrand[lc200_oob] = 0. # PREVENTS CATASTROPHIC FAILURE DUE TO FINITE C200 GRID

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

output = h5py.File('nfw_shmr_inference.hdf5', 'w')
output.create_dataset('logp', data=sampler.lnprobability)
for n in range(npars):
    output.create_dataset(pars[n]['name'], data=sampler.chain[:, :, n])

