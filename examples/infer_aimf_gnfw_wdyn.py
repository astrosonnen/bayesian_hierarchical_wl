import numpy as np
from scipy.stats import truncnorm
from scipy.interpolate import splrep, splev
from scipy.special import erf
import emcee
import h5py
import sys, os
import ndinterp


mockname = 'isolated_gnfw_fixedz_wdyn'

mock = h5py.File('../wl_sims/%s_mock.hdf5'%mockname, 'r')

griddir = '/net/ringvaart/data2/sonnenfeld/4hs_forecasts/%s_mock_grids/'%mockname
dyndir = '/net/ringvaart/data2/sonnenfeld/4hs_forecasts/'
outdir = '/net/ringvaart/data2/sonnenfeld/4hs_forecasts/'

nstep = 100
nwalkers = 50
nis = 1000

nlens = mock.attrs['nlens']
lmsps_piv = mock.attrs['lmsps_piv']
lmsps_err = mock.attrs['lmsps_err']
lmsps_mu = mock.attrs['lmsps_mu']
lmsps_sig = mock.attrs['lmsps_sig']
#vdisp_err = mock.attrs['vdisp_err']
vdisp_err = 10. # to be changed. need to re-run sample generation script

lreff_mu = mock.attrs['lreff_mu']
lreff_sig = mock.attrs['lreff_sig']
lreff_beta = mock.attrs['lreff_beta']

gammadm_min = mock.attrs['gammadm_min']
gammadm_max = mock.attrs['gammadm_max']

grids = []

lmsps_impsamp = []
lm200_impsamp = []
gammadm_impsamp = []

lmsps_obs = []
vdisp_obs = []
lreff_samp = []

lmstar_grids = []
lm200_grids = []
gammadm_grids = []

# loads dynamics grid
dyngrid_file = h5py.File(dyndir+'/%s_dyngrid.hdf5'%mockname, 'r')
lreff_dyngrid = dyngrid_file['lreff_grid'][()]
lm200_dyngrid = dyngrid_file['lm200_grid'][()]
gammadm_dyngrid = dyngrid_file['gammadm_grid'][()]
s2_gnfw_dyngrid = dyngrid_file['s2_gnfw_grid'][()]
s2_deV_dyngrid = dyngrid_file['s2_deV_grid'][()]

s2_deV_spline = splrep(lreff_dyngrid, s2_deV_dyngrid)

s2_deV_samp = []
gnfw_dynaxes = {0: splrep(lm200_dyngrid, np.arange(len(lm200_dyngrid))), 1: splrep(gammadm_dyngrid, np.arange(len(gammadm_dyngrid))), 2: splrep(lreff_dyngrid, np.arange(len(lreff_dyngrid)))}

s2_gnfw_dyninterp = ndinterp.ndInterp(gnfw_dynaxes, s2_gnfw_dyngrid, order=3)

for i in range(nlens):

    lensname = 'lens_%05d'%i
    gridname = griddir+'/%s_wl_likegrid.hdf5'%lensname

    if os.path.isfile(gridname):

        grid_file = h5py.File(gridname, 'r')

        lmstar_grid = grid_file['mstar_grid'][()]
        lm200_grid = grid_file['m200_grid'][()]
        gammadm_grid = grid_file['gammadm_grid'][()]

        lmstar_grids.append(lmstar_grid)
        lm200_grids.append(lm200_grid)
        gammadm_grids.append(gammadm_grid)

        axes = {0: splrep(lmstar_grid, np.arange(len(lmstar_grid))), 1: splrep(lm200_grid, np.arange(len(lm200_grid))), 2: splrep(gammadm_grid, np.arange(len(gammadm_grid)))}

        grid_here = grid_file['wl_like_grid'][()]
        grid_here -= grid_here.max()

        grid = ndinterp.ndInterp(axes, grid_here, order=1)
        grids.append(grid)

        grid_file.close()

        lmsps_sig_eff = (1./lmsps_sig**2 + lreff_beta**2/lreff_sig**2)**(-0.5)

        lmsps_mu_given_lreff = (mock['lreff_samp'][i] - lreff_mu)/lreff_beta + lmsps_piv
        lmsps_mu_eff = (lmsps_mu/lmsps_sig**2 + lmsps_mu_given_lreff/(lreff_sig/lreff_beta)**2)*lmsps_sig_eff**2

        lmsps_here = np.random.normal(lmsps_mu_eff, lmsps_sig_eff, nis)
        lmsps_impsamp.append(lmsps_here)
 
        lm200_impsamp.append(np.random.normal(0., 1., 1000))

        gammadm_impsamp.append(np.random.normal(0., 1., 1000))

        lmsps_obs.append(mock['lmsps_samp'][i])
        vdisp_obs.append(mock['vdisp_obs'][i])
        lreff_samp.append(mock['lreff_samp'][i])

        # dynamics
        s2_deV_samp.append(splev(mock['lreff_samp'][i], s2_deV_spline))

mock.close()

nsamp = len(grids)

lm200_mu = {'name': 'lm200_mu', 'lower': 11., 'upper': 15., 'guess': 12.8, 'step': 0.1}
lm200_sig = {'name': 'lm200_sig', 'lower': 0., 'upper': 1., 'guess': 0.3, 'step': 0.03}
lm200_lmsps_dep = {'name': 'lm200_msps_dep', 'lower': -3., 'upper': 3., 'guess': 1.7, 'step': 0.1}
gammadm_mu = {'name': 'gammadm_mu', 'lower': gammadm_min, 'upper': gammadm_max, 'guess': 1.3, 'step': 0.1}
gammadm_sig = {'name': 'gammadm_sig', 'lower': 0., 'upper': 1., 'guess': 0.1, 'step': 0.03}
laimf = {'name': 'laimf', 'lower': 0., 'upper': 0.3, 'guess': 0.1, 'step': 0.1}

pars = [lm200_mu, lm200_sig, lm200_lmsps_dep, gammadm_mu, gammadm_sig, laimf]

npars = len(pars)

bounds = []
for par in pars:
    bounds.append((par['lower'], par['upper']))

def logprior(p):
    for i in range(npars):
        if p[i] < bounds[i][0] or p[i] > bounds[i][1]:
            return -1e300
    return 0.

def logpfunc(p):

    lprior = logprior(p)
    if lprior < 0.:
        return -1e300

    lm200_mu, lm200_sig, lm200_lmsps_dep, gammadm_mu, gammadm_sig, laimf = p

    sumlogp = 0.

    for i in range(nsamp):

        lm200_here = lm200_mu + lm200_lmsps_dep * (lmsps_impsamp[i] - lmsps_piv) + lm200_impsamp[i] * lm200_sig
        lmstar_here = lmsps_impsamp[i] + laimf

        gammadm_here = gammadm_mu + gammadm_sig * gammadm_impsamp[i]

        good = (lm200_here > lm200_grids[i][0]) & (lm200_here < lm200_grids[i][-1]) & (lmstar_here > lmstar_grids[i][0]) & (lmstar_here < lmstar_grids[i][-1]) & (gammadm_here > gammadm_grids[i][0]) & (gammadm_here < gammadm_grids[i][-1])

        point = np.array((lmstar_here, lm200_here, gammadm_here)).T

        wl_loglike = grids[i].eval(point)

        # dynamics
        s2_stars = 10.**lmstar_here * s2_deV_samp[i]

        dynpoint = np.array((lm200_here, gammadm_here, lreff_samp[i]))
        s2_halo = s2_gnfw_dyninterp.eval(dynpoint)

        vdisp_here = (s2_stars + s2_halo)**0.5
        vdisp_loglike = -0.5*(vdisp_here - vdisp_obs[i])**2/vdisp_err**2

        lmsps_loglike = -0.5*(lmsps_impsamp[i] - lmsps_obs[i])**2/lmsps_err**2

        integrand = np.exp(wl_loglike) * np.exp(lmsps_loglike) * np.exp(vdisp_loglike)

        sumlogp += np.log(integrand[good].mean())

    if sumlogp != sumlogp:
        return -1e300

    return sumlogp

sampler = emcee.EnsembleSampler(nwalkers, npars, logpfunc, threads=50)

start = []
if len(sys.argv) > 1:
    print('using last step of %s to initialize walkers'%sys.argv[1])
    startfile = h5py.File('%s'%sys.argv[1], 'r')

    for i in range(nwalkers):
        tmp = np.zeros(npars)
        for n in range(npars):
            tmp[n] = startfile[pars[n]['name']][i, -1]
        start.append(tmp)
    startfile.close()
else:
    for i in range(nwalkers):
        tmp = np.zeros(npars)
        for j in range(npars):
            a, b = (bounds[j][0] - pars[j]['guess'])/pars[j]['step'], (bounds[j][1] - pars[j]['guess'])/pars[j]['step']
            p0 = truncnorm.rvs(a, b, size=1)*pars[j]['step'] + pars[j]['guess']
            tmp[j] = p0

        start.append(tmp)

print("Sampling on %d galaxies"%(len(grids)))

sampler.run_mcmc(start, nstep)

output = h5py.File(outdir+'/%s_aimf_gnfw_wdyn_inference.hdf5'%mockname, 'w')
output.create_dataset('logp', data=sampler.lnprobability)
for n in range(npars):
    output.create_dataset(pars[n]['name'], data=sampler.chain[:, :, n])

