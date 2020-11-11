import numpy as np
from wl_profiles import gnfw, sersic
import wl_lens_models
from wl_cosmology import Mpc, c, G, M_Sun
import wl_cosmology
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import truncnorm
from scipy.interpolate import splrep, splev
import h5py


# N lenses, gNFW dark matter halo + de Vaucouleurs stellar bulge
# log-Normal distribution in stellar mass
# All lenses at the same redshift, all sources at the same redshift.
# Fixed concentration.

mockname = 'isolated_gnfw_fixedz_mock'

nlens = 10000 # sample size

np.random.seed(0)

lmsps_piv = 11.5 # pivot point of stellar mass-halo mass and stellar mass-size relation
lmsps_err = 0.15 # uncertainty on logM* from stellar population synthesis
vdisp_err = 10. # uncertainty on velocity dispersion (in km/s)

zd = 0.2 # lens redshift
zs = 1. # source redshift
c200 = 5. # halo concentration

kpc = Mpc/1000.
arcsec2rad = np.deg2rad(1./3600.)

dd = wl_cosmology.Dang(zd)
ds = wl_cosmology.Dang(zs)
dds = wl_cosmology.Dang(zs, zd)

rhoc = wl_cosmology.rhoc(zd) # critical density of the Universe at z=zd. Halo masses are defined as M200 wrt rhoc.

arcsec2kpc = arcsec2rad * dd * 1000.

fiber_radius = 1. # radius of spectroscopic fiber (in arcsec)
fiber_radius_kpc = fiber_radius * arcsec2kpc

# source distribution parameters
sigma_eps = 0.25 # intrinsic scatter in shape distribution
nbkg = 25. # source number density (in arcmin^-2)
Rmin_Mpc = 0.03 # minimum radius for shape measurements (in Mpc)
Rmax_Mpc = 0.3 # maximum radius for shape measurements (in Mpc)

Rmin_deg = np.rad2deg(Rmin_Mpc/dd)
Rmax_deg = np.rad2deg(Rmax_Mpc/dd)

nsource_avg = nbkg * np.pi * (Rmax_deg**2 - Rmin_deg**2) * 3600. # average number of sources per lens

Rfrac_min = gnfw.R_grid[0] # lower bound of R in gNFW profile grids, in units of r_s
Rfrac_max = gnfw.R_grid[-1] # lower bound of R in gNFW profile grids, in units of r_s

lmsps_mu = 11.4 # average value of logM*^(sps)
lmsps_sig = 0.3 # intrinsic scatter in logM*^(sps)

laimf_mu = 0.1 # average value of log(alpha_IMF)
laimf_sig = 0. # intrinsic scatter in log(alpha_IMF)

lreff_mu = 1. # average value of log(Reff) at logM*=lmsps_piv
lreff_beta = 0.8 # slope of mass-size relation
lreff_sig = 0.15 # intrinsic scatter in Reff at fixed logM*
nser = 4. # Sersic index (same for all)

lm200_mu = 13. # average logM200 at logM*=lmsps_piv
lm200_sig = 0.2 # intrinsic scatter in logM200
lm200_beta = 1.5 # slope of stellar mass-halo mass relation

gammadm_mu = 1.3 # average inner slope of dark matter density profile
gammadm_sig = 0.1 # intrinsic scatter on the inner dark matter slope

gammadm_min = 0.8 # smallest allowed value of the dark matter inner slope
gammadm_max = 1.6 # largest allowed value of the dark matter inner slope

# generate the values of stellar mass, size, IMF, halo mass
lmsps_samp = np.random.normal(lmsps_mu, lmsps_sig, nlens)

lreff_samp = lreff_mu + lreff_beta * (lmsps_samp - lmsps_piv) + np.random.normal(0., lreff_sig, nlens)

lm200_samp = lm200_mu + lm200_beta * (lmsps_samp - lmsps_piv) + np.random.normal(0., lm200_sig, nlens)

laimf_samp = laimf_mu + laimf_sig * np.random.normal(0., 1., nlens)
lmstar_samp = lmsps_samp + laimf_samp

# adds observational errors to the stellar mass measurements
lmsps_obs = lmsps_samp + np.random.normal(0., lmsps_err, nlens)

# generates observational errors on the central velocity dispersion
vdisp_delta = np.random.normal(0., vdisp_err, nlens)

# generates values of the inner dark matter slope
gammadm_a, gammadm_b = (gammadm_min - gammadm_mu)/gammadm_sig, (gammadm_max - gammadm_mu)/gammadm_sig
gammadm_samp = truncnorm.rvs(gammadm_a, gammadm_b, size=nlens)*gammadm_sig + gammadm_mu

# draws the number of sources from a Poisson distribution
nsource_samp = np.random.poisson(lam=nsource_avg, size=nlens)

# stores everything generated so far into an .hdf5 file
output = h5py.File('%s.hdf5'%mockname, 'w')

# individual lens parameters
output.create_dataset('lmsps_samp', data=lmsps_samp)
output.create_dataset('lmsps_obs', data=lmsps_obs)
output.create_dataset('lmstar_samp', data=lmstar_samp)
output.create_dataset('laimf_samp', data=laimf_samp)
output.create_dataset('lm200_samp', data=lm200_samp)
output.create_dataset('gammadm_samp', data=gammadm_samp)
output.create_dataset('lreff_samp', data=lreff_samp)
output.create_dataset('nsource_samp', data=nsource_samp)

output.attrs['nlens'] = nlens
output.attrs['zd'] = zd
output.attrs['zs'] = zs
output.attrs['c200'] = c200
output.attrs['gammadm_max'] = gammadm_max
output.attrs['gammadm_min'] = gammadm_min
output.attrs['nser'] = nser
output.attrs['nbkg'] = nbkg
output.attrs['sigma_eps'] = sigma_eps

# hyper-parameters
output.attrs['lmsps_mu'] = lmsps_mu
output.attrs['lmsps_sig'] = lmsps_sig
output.attrs['lmsps_err'] = lmsps_err
output.attrs['lmsps_piv'] = lmsps_piv
output.attrs['laimf_mu'] = laimf_mu
output.attrs['laimf_sig'] = laimf_sig
output.attrs['lm200_mu'] = lm200_mu
output.attrs['lm200_sig'] = lm200_sig
output.attrs['lm200_beta'] = lm200_beta
output.attrs['gammadm_mu'] = gammadm_mu
output.attrs['gammadm_sig'] = gammadm_sig
output.attrs['lreff_mu'] = lreff_mu
output.attrs['lreff_sig'] = lreff_sig
output.attrs['lreff_beta'] = lreff_beta

lens_model = wl_lens_models.GNFWdeV(z=zd, c200=c200)

s_cr = lens_model.S_cr(zs) # lensing critical surface mass density (in M_Sun/Mpc^2)

for i in range(nlens):
    print(i)

    # source position: uniform distribution in an annulus between Rmin and Rmax
    tsource = np.random.rand(nsource_samp[i])
    Rsource_deg = (tsource * (Rmax_deg**2 - Rmin_deg**2) + Rmin_deg**2)**0.5

    lens_model.m200 = 10.**lm200_samp[i]
    lens_model.reff = 10.**lreff_samp[i]
    lens_model.mstar = 10.**lmstar_samp[i]
    lens_model.gammadm = gammadm_samp[i]

    lens_model.update()

    # calculates reduced shear and adds shape noise
    gammat_source = lens_model.gammat(Rsource_deg, s_cr)
    kappa_source = lens_model.kappa(Rsource_deg, s_cr)
    g_source = gammat_source/(1.-kappa_source)
    et_obs = g_source + np.random.normal(0., sigma_eps, nsource_samp[i])

    r200 = (10.**lm200_samp[i]*3./200./(4.*np.pi)/rhoc)**(1./3.) * 1000.

    group = output.create_group('lens_%05d'%i)

    group.create_dataset('R_source', data=Rsource_deg)
    group.create_dataset('g_source', data=g_source)
    group.create_dataset('et_obs', data=et_obs)

