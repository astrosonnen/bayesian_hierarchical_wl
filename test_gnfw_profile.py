import pylab
import numpy as np
import wl_lens_models
import lens_tools
from wl_profiles import gnfw


zd = 0.3
m200 = 1e13
mstar = 1e11
reff = 1.
c200 = 5.
gamma = 1.2

zs = 1.

nfw_lens = wl_lens_models.NFWPoint(z=zd, m200=m200, mstar=mstar, c200=c200)
gnfw_lens = wl_lens_models.GNFWPoint(z=zd, m200=m200, mstar=mstar, c200=c200, gamma=gamma)

nr = 101
r_arr = np.logspace(-2., 0., nr) * nfw_lens.Mpc2deg

nfw_sources = {}
nfw_sources['r'] = np.random.rand(nr) * (r_arr[-1] - r_arr[0]) + r_arr[0]
nfw_sources['z'] = zs * np.ones(nr) # np.random.rand(nr) + 0.5 + zd

gnfw_sources = {}
gnfw_sources['r'] = np.random.rand(nr) * (r_arr[-1] - r_arr[0]) + r_arr[0]
gnfw_sources['z'] = zs * np.ones(nr) # np.random.rand(nr) + 0.5 + zd

nfw_gamma = 0. * r_arr
gnfw_gamma = 0. * r_arr

nfw_kappa = 0. * r_arr
gnfw_kappa = 0. * r_arr

for i in range(nr):
    nfw_gamma[i] = nfw_lens.gammat(r_arr[i], zs)
    gnfw_gamma[i] = gnfw_lens.gammat(r_arr[i], zs)

    nfw_kappa[i] = nfw_lens.kappa(r_arr[i], zs)
    gnfw_kappa[i] = gnfw_lens.kappa(r_arr[i], zs)

nfw_lens.sources = nfw_sources
nfw_lens.get_source_scr()

gnfw_lens.sources = gnfw_sources
gnfw_lens.get_source_scr()

nfw_lens.get_source_gammat()
nfw_lens.get_source_kappa()

gnfw_lens.get_source_gammat()
gnfw_lens.get_source_kappa()

pylab.subplot(1, 2, 1)
pylab.loglog(r_arr, nfw_gamma)
pylab.loglog(r_arr, gnfw_gamma, linestyle='--')
pylab.scatter(nfw_sources['r'], nfw_lens.sources['gammat'], marker='x')
pylab.scatter(gnfw_sources['r'], gnfw_lens.sources['gammat'], marker='.')

pylab.subplot(1, 2, 2)
pylab.loglog(r_arr, nfw_kappa)
pylab.loglog(r_arr, gnfw_kappa, linestyle='--')
pylab.scatter(nfw_sources['r'], nfw_lens.sources['kappa'], marker='x')
pylab.scatter(gnfw_sources['r'], gnfw_lens.sources['kappa'], marker='.')

pylab.show()



"""
gamma = (0.55, 1., 1.45)

for g in gamma:

    num_m2d = 0. * r_arr
    fast_m2d = 0. * r_arr

    num_m3d = 0. * r_arr
    fast_m3d = 0. * r_arr

    norm = m200 / gnfw.M3d(nfw_lens.r200, nfw_lens.rs, g)

    for i in range(nr):
        num_m2d[i] = norm * gnfw.M2d(r_arr[i]/nfw_lens.Mpc2deg, nfw_lens.rs, g)
        fast_m2d[i] = norm * gnfw.fast_M2d(r_arr[i]/nfw_lens.Mpc2deg, nfw_lens.rs, g)

        num_m3d[i] = norm * gnfw.M3d(r_arr[i]/nfw_lens.Mpc2deg, nfw_lens.rs, g)
        fast_m3d[i] = norm * gnfw.fast_M3d(r_arr[i]/nfw_lens.Mpc2deg, nfw_lens.rs, g)

    pylab.subplot(1, 2, 1)
    pylab.loglog(r_arr/nfw_lens.Mpc2deg, num_m2d)
    pylab.loglog(r_arr/nfw_lens.Mpc2deg, fast_m2d, linestyle='--')

    pylab.subplot(1, 2, 2)
    pylab.loglog(r_arr/nfw_lens.Mpc2deg, num_m3d)
    pylab.loglog(r_arr/nfw_lens.Mpc2deg, fast_m3d, linestyle='--')

    pylab.show()
"""

