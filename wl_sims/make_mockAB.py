# these functions generate mock data
import numpy as np
from scipy.stats import truncnorm
from scipy.interpolate import splrep, splev, splint
import h5py
from cosmolopy import distance


mockname = 'mockAB'

cosmo = {'omega_M_0': omegaM, 'omega_lambda_0': omegaL, 'omega_k_0': omegak, 'h': h}

nbkg = 20 # number density of background sources (arcmin^-2)
zd_mu = 0.2 # average lens redshift
zd_sig = 0.1 # lens redshift dispersion
zd_min = 0.1 # minimum lens redshift

zs_mu = 1. # average source redshift
zs_sig = 0.5 # source redshift dispersion
zs_err = 0.1 # uncertainty on the source redshift

nz = 101

sigma_eps = 0.27 # dispersion in intrinsic source shape

rmax = 1. # maximum radius (in Mpc) out to which sources are drawn

mstar_mu = 11. # average logM*
mstar_sig = 0.4 # dispersion in logM*

mstar_err = 0.15 # uncertainty on logM*

mstar_cut = 11. # minimum cut on logM*(obs)

mstar_piv = 11.2 # pivot stellar mass

# Halo mass distribution: Gaussian in logMh, with mean
# mu_h = mhalo_mu + beta_h * (mhalo_mu - mstar_piv)
mhalo_mu = 13. # average logMh at the pivot stellar mass
mhalo_sig = 0.4 # intrinsic scatter in logMh around the mean
beta_h = 1.5 # halo mass-stellar mass correlation

# size distribution
re_beta = 0.57 # slope of mstar-Reff relation
re_mu = 0.54 + 0.25*re_beta # average logRe at logM*=11 (after changing IMF)
re_sig = 0.16 # intrinsic scatter in logRe
re_err = 0.04 # uncertainty on logRe

n0 = 1000000 # large number for random sample drawing purposes

nlens = 5000 # number of lenses

dec_max = 45. # maximum declination (in case coordinates are needed)

# first draws true stellar masses from a Gaussian
mstar = np.random.normal(mstar_mu, mstar_sig, n0)

# applies observational error
mstar_obs = mstar + np.random.normal(0., mstar_err, n0)

# applies cut on observed stellar mass
sel = mstar_obs > mstar_cut

indices = np.arange(n0)
# only keeps the first nlens lenses
samp = np.random.choice(indices[sel], nlens) 

mstar = mstar[samp]
mstar_obs = mstar_obs[samp]

# draws halo masses
mhalo = mhalo_mu + beta_h * (mstar - mstar_piv) + np.random.normal(0., mhalo_sig, nlens)

# draws sizes
reff = re_mu + re_beta * (mstar - 11.) + np.random.normal(0., re_sig, nlens)
# adds observational error to sizes
reff_obs = reff + np.random.normal(0., re_err, nlens)

# draws concentration
lc200 = c200_mu + c200_beta * (mhalo - c200_piv) + np.random.normal(0., c200_sig, nlens)

# draws lens redshift (one-side truncated Gaussian)
a, b = (zd_min - zd_mu)/zd_sig, (np.inf - zd_mu)/zd_sig
zd = truncnorm.rvs(a, b, size=nlens)*zd_sig + zd_mu

# draws coordinates
ra = 360. * np.random.rand(nlens)
dec = np.rad2deg(np.arcsin(np.sin(np.deg2rad(dec_max))*(np.random.rand(nlens)*2. - 1.)))

# stores data
hdf5_file = h5py.File(mockname+'_pop.hdf5', 'w')

hyperpars = {'mstar_mu': mstar_mu, 'mstar_sig': mstar_sig, 'mstar_err': mstar_err, 'mstar_cut': mstar_cut, \
             'mhalo_mu': mhalo_mu, 'mhalo_sig': mhalo_sig, 'beta_h': beta_h, 're_beta': re_beta, 're_mu': re_mu, \
             're_sig': re_sig, 're_err': re_err}

pop = {'mstar': mstar, 'mstar_obs': mstar_obs, 'mhalo': mhalo, 'c200': lc200, \
       'reff': 10.**reff, 'reff_obs': 10.**reff_obs, 'ra': ra, 'dec': dec, 'zd': zd}

cosmo_group = hdf5_file.create_group('cosmo')
for par in cosmo:
    cosmo_group.create_dataset(par, data=cosmo[par])

for par in pop:
    hdf5_file.create_dataset(par, data=pop[par])
    
for par in hyperpars:
    hdf5_file.create_dataset(par, data=hyperpars[par])

hdf5_file.close()

# now creates sources

print 'creating sources...'

source_file = h5py.File('%s_sources.hdf5'%mockname, 'w')

source_file.create_dataset('rmax', data=rmax)
source_file.create_dataset('zs_err', data=zs_err)
source_file.create_dataset('sigma_eps', data=sigma_eps)
source_file.create_dataset('nbkg', data=nbkg)

for i in range(nlens):
    print i
    
    Dd = distance.angular_diameter_distance(zd[i], **cosmo)
    rmax_deg = np.rad2deg(rmax/Dd)
    
    area = pi*(rmax_deg*60.)**2  # area of circle with radius rmax
    nsource = np.random.poisson(area*nbkg) # draws the number of sources

    # draws source positions (uniform in the circle)
    r_source = (rmax_deg**2*np.random.rand(nsource))**0.5
    phi_source = 2.*pi*np.random.rand(nsource)

    ra_source = ra[i] - r_source*np.cos(phi_source)/np.cos(np.deg2rad(dec[i]))
    dec_source = dec[i] + r_source*np.sin(phi_source)

    # draws redshifts
    a, b = -zs_mu/zs_sig, (5. - zs_mu)/zs_sig
    zs = truncnorm.rvs(a, b, size=nsource)*zs_sig + zs_mu
    
    # adds errors to the redshifts
    zs_obs = zs + np.random.normal(0., zs_err, nsource)
    zs_obs[zs_obs < 0.] = 0.
   
    # draws the intrinsic ellipticity: Gaussian with dispersion sigma_eps
    eps11s = np.random.normal(0., sigma_eps, nsource)
    eps12s = np.random.normal(0., sigma_eps, nsource)

    # complex intrinsic ellipticity
    es = eps11s + 1j*eps12s

    # calculates the critical density of each source
    s_cr = 0.*zs
    s_cr_obs = 0.*zs

    bkg = zs > zd[i]
    fgd = np.logical_not(bkg)

    obs_bkg = zs_obs > zd[i]

    zs_grid = np.logspace(np.log10(min(zs[bkg].min(), zs_obs[obs_bkg].min())), np.log10(max(zs.max(), zs_obs[obs_bkg].max() + zs_err)), nz)

    s_cr_grid = 0.*zs_grid

    for j in range(nz):
        Ds = distance.angular_diameter_distance(zs_grid[j], **cosmo)
        Dds = distance.angular_diameter_distance(zs_grid[j], zd[i], **cosmo)
        s_cr_grid[j] = c**2/(4.*pi*G)*Ds/Dds/Dd*Mpc/M_Sun

    s_cr_spline = splrep(zs_grid, s_cr_grid, k=1)

    s_cr[bkg] = splev(zs[bkg], s_cr_spline)
    s_cr_obs[obs_bkg] = splev(zs_obs[obs_bkg], s_cr_spline)

    """
    for j in range(nsource):
        if bkg[j]:
            Ds = distance.angular_diameter_distance(zs[j], **cosmo)
            Dds = distance.angular_diameter_distance(zs[j], zd[i], **cosmo)
            s_cr[j] = c**2/(4.*pi*G)*Ds/Dds/Dd*Mpc/M_Sun

    s_cr_obs_up = 0.*s_cr_obs

    for j in range(nsource):
        if obs_bkg[j]:
            Ds = distance.angular_diameter_distance(zs_obs[j], **cosmo)
            Dds = distance.angular_diameter_distance(zs_obs[j], zd[i], **cosmo)
            s_cr_obs[j] = c**2/(4.*pi*G)*Ds/Dds/Dd*Mpc/M_Sun

            Dsup = distance.angular_diameter_distance(zs_obs[j] + zs_err, **cosmo)
            Ddsup = distance.angular_diameter_distance(zs_obs[j] + zs_err, zd[i], **cosmo)
            s_cr_obs_up[j] = c**2/(4.*pi*G)*Dsup/Ddsup/Dd*Mpc/M_Sun

    s_cr_err = s_cr_obs[obs_bkg] - s_cr_obs_up[obs_bkg]
    """

    # propagates the redshift error to the critical density
    s_cr_obs_up = splev(zs_obs[obs_bkg] + zs_err, s_cr_spline)
    s_cr_err = s_cr_obs[obs_bkg] - s_cr_obs_up

    # shape measurement error: 
    # quadrature sum of intrinsic scatter
    shape_var = sigma_eps**2 * np.ones(nsource)
    # and error on Sigma_cr
    shape_var[obs_bkg] += (s_cr_err/s_cr_obs[obs_bkg])**2

    sources = {}
    sources['ra'] = ra_source
    sources['dec'] = dec_source
    sources['z'] = zs
    sources['z_obs'] = zs_obs
    sources['e1s'] = eps11s
    sources['e2s'] = eps12s
    sources['s_cr'] = s_cr
    sources['s_cr_obs'] = s_cr_obs
    sources['s_cr_err'] = s_cr_err
    sources['r'] = r_source
    sources['phi'] = phi_source
    sources['weight'] = 1./shape_var

    source_group = source_file.create_group('lens_%05d'%i)
    for par in sources:
        source_group.create_dataset(par, data=sources[par], dtype='f')

source_file.close()

