import numpy as np
import h5py
import wl_cosmology
from wl_cosmology import c, G, Mpc, M_Sun
from scipy.interpolate import splrep, splev
from astropy.io import fits as pyfits


calib_dir = '/Users/sonnen/hsc_weaklensing/hsc-unblinded-Aug2017/'

# reads in the shear calibration info
calib_file = h5py.File(calib_dir+'hsc_16a_mandelbaum_shear.hdf5', 'r')

rmax = 0.5 # maximum radius for the WL analysis, in Mpc (physical)
rmin = 0.03 # minimum radius for the WL analysis, in Mpc (physical)

nz = 1001
zd_grid = np.linspace(0.01, 1., nz)
dd_grid = 0.*zd_grid
for i in range(nz):
    dd_grid[i] = wl_cosmology.Dang(zd_grid[i])

dd_spline = splrep(zd_grid, dd_grid)

# reads in the shape measurement and photo-z table
wl_table = h5py.File('/gdrive/projects/hsc_weaklensing/HSC_S16A_2.0_minimal.hdf5', 'r') 

# reads in the lens sample catalog
f = open('sdss_legacy_hscoverlap_mcut11.0.cat', 'r')
ra, dec, zd, mstar, mstar_err = np.loadtxt(f, usecols=(0, 1, 2, 3, 4), unpack=True)
f.close()

ngal = len(ra)

theta_max = np.rad2deg(rmax / splev(zd, dd_spline))
theta_min = np.rad2deg(rmin / splev(zd, dd_spline))

def get_wl_field(ra, dec):

    if (ra > 213.) & (ra < 216.) & (dec > 50.) & (dec < 54.):
        return 'AEGIS'

    elif (ra > 128.) & (ra < 142.) & (dec > -2.) & (dec < 5.):
        return 'GAMA09H'

    elif (ra > 210.) & (ra < 226.) & (dec > -2.) & (dec < 2.):
        return 'GAMA15H'

    elif (ra > 236.) & (ra < 249.) & (dec > 42.) & (dec < 45.):
        return 'HECTOMAP'

    elif (ra > 330.) & (ra < 342.) & (dec > -1.) & (dec < 3.):
        return 'VVDS'
    
    elif (ra > 175.) & (ra < 183.) & (dec > -2.) & (dec < 2.):
        return 'WIDE12H'

    elif (ra > 29.) & (ra < 40.) & (dec > -7.) & (dec < -1.):
        return 'XMM'

    else:
        return None

source_file = h5py.File('camira_specz_sources.hdf5', 'w')

for i in range(ngal):

    field = get_wl_field(ra[i], dec[i])

    print i, field

    if field is not None:

        group = source_file.create_group('lens_%04d'%i)

        source_dists = ((ra[i] - wl_table[field]['ira'].value)**2*np.cos(np.deg2rad(dec[i]))**2 + (dec[i] - wl_table[field]['idec'].value)**2)**0.5
        
        keep = (source_dists < theta_max[i]) & (source_dists > theta_min[i]) & (wl_table[field]['mizuki_photoz_err95_min'].value > zd[i])
    
        group.create_dataset('r', data=source_dists[keep])
        group.create_dataset('z', data=wl_table[field]['mizuki_photoz_median'][keep])
        group.create_dataset('z_16', data=wl_table[field]['mizuki_photoz_err68_min'][keep])
        group.create_dataset('z_84', data=wl_table[field]['mizuki_photoz_err68_max'][keep])
        group.create_dataset('rms_e', data=calib_file[field]['ishape_hsm_regauss_derived_rms_e'][keep])
        group.create_dataset('sigma_e', data=calib_file[field]['ishape_hsm_regauss_derived_sigma_e'][keep])
        group.create_dataset('m_bias', data=calib_file[field]['ishape_hsm_regauss_derived_shear_bias_m'][keep])
    
        sel_e1 = wl_table[field]['ishape_hsm_regauss_e1'][keep]
        sel_e2 = -wl_table[field]['ishape_hsm_regauss_e2'][keep]
    
        sel_c1 = calib_file[field]['ishape_hsm_regauss_derived_shear_bias_c1'][keep]
        sel_c2 = -calib_file[field]['ishape_hsm_regauss_derived_shear_bias_c2'][keep]
    
        sel_ra = wl_table[field]['ira'][keep]
        sel_dec = wl_table[field]['idec'][keep]
     
        dra = (sel_ra - ra[i])/np.cos(np.deg2rad(dec[i]))
        ddec = sel_dec - dec[i]
    
        phis = np.arctan(dra/ddec)
        phis[ddec<0.] = phis[ddec<0.] + np.pi
        phis[phis<0.] += 2.*np.pi
    
        group.create_dataset('phi', data=phis)
    
        cosphi = np.cos(phis)
        sinphi = np.sin(phis)
    
        sin2phi = 2.*cosphi*sinphi
        cos2phi = cosphi**2 - sinphi**2
    
        group.create_dataset('et', data=sel_e1*cos2phi + sel_e2*sin2phi)
        group.create_dataset('er', data=-sel_e1*sin2phi + sel_e2*cos2phi)
    
        group.create_dataset('ct_bias', data=sel_c1*cos2phi + sel_c2*sin2phi)
        group.create_dataset('cr_bias', data=-sel_c1*sin2phi + sel_c2*cos2phi)
     
        # calculates critical densities
    
        nsource = keep.sum()
        ds = np.zeros(nsource)
        dds = np.zeros(nsource)
    
        s_cr = np.zeros(nsource)
    
        dd = wl_cosmology.Dang(zd[i])
    
        for j in range(nsource):
            ds[j] = wl_cosmology.Dang(group['z'][j])
            dds[j] = wl_cosmology.Dang(zd[i], group['z'][j])
    
        s_cr = c**2/(4.*np.pi*G)*ds/dds/dd*Mpc/M_Sun
    
        group.create_dataset('s_cr', data=s_cr)

