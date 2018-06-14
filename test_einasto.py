import pylab
import numpy as np
import wl_lens_models
import lens_tools


zd = 0.3
m200 = 10.**13.7
mstar = 1e11
c200 = 5.
alpha = 0.28
#ein_c200 = 10.**0.45
ein_c200 = 5.

zs = 1.

nfw_lens = wl_lens_models.NFWPoint(z=zd, m200=m200, mstar=mstar, c200=c200)
ein_lens = wl_lens_models.EinastoPoint(z=zd, m200=m200, mstar=mstar, c200=ein_c200, alpha=alpha)

nr = 101
r_arr = np.logspace(-2., 0., nr) * nfw_lens.Mpc2deg

sources = {}
sources['r'] = np.random.rand(nr) * (r_arr[-1] - r_arr[0]) + r_arr[0]
sources['z'] = np.random.rand(nr) + 0.5 + zd

nfw_gamma = 0. * r_arr
ein_gamma = 0. * r_arr

nfw_kappa = 0. * r_arr
ein_kappa = 0. * r_arr

nfw_ds = 0. * r_arr
ein_ds = 0. * r_arr

nfw_Sigmabar = 0. * r_arr
ein_Sigmabar  = 0. * r_arr

for i in range(nr):

    nfw_gamma[i] = nfw_lens.gammat(r_arr[i], zs)
    ein_gamma[i] = ein_lens.gammat(r_arr[i], zs)

    nfw_kappa[i] = nfw_lens.kappa(r_arr[i], zs)
    ein_kappa[i] = ein_lens.kappa(r_arr[i], zs)

    nfw_ds[i] = nfw_lens.DeltaSigma(r_arr[i])
    ein_ds[i] = ein_lens.DeltaSigma(r_arr[i])

    nfw_Sigmabar[i] = nfw_lens.Sigmabar(r_arr[i])
    ein_Sigmabar[i] = ein_lens.Sigmabar(r_arr[i])

pylab.subplot(1, 2, 1)
pylab.loglog(r_arr, nfw_gamma)
pylab.loglog(r_arr, ein_gamma, linestyle='--')

pylab.subplot(1, 2, 2)
pylab.loglog(r_arr, nfw_kappa)
pylab.loglog(r_arr, ein_kappa, linestyle='--')

pylab.show()

nfw_lens.sources = sources
nfw_lens.get_source_scr()

ein_lens.sources = sources
ein_lens.get_source_scr()

nfw_lens.get_source_kappa()
nfw_lens.get_source_gammat()

ein_lens.get_source_kappa()
ein_lens.get_source_gammat()

pylab.subplot(1, 2, 1)

pylab.scatter(nfw_lens.sources['gammat'], ein_lens.sources['gammat'], marker='x')
xlim = pylab.xlim()
xs = np.linspace(xlim[0], xlim[1])
pylab.plot(xs, xs, linestyle=':', color='k')

pylab.subplot(1, 2, 2)

pylab.scatter(nfw_lens.sources['kappa'], ein_lens.sources['kappa'], marker='x')
xlim = pylab.xlim()
xs = np.linspace(xlim[0], xlim[1])
pylab.plot(xs, xs, linestyle=':', color='k')

pylab.show()

pylab.subplot(1, 2, 1)

pylab.loglog(r_arr, nfw_ds, linestyle='--')
pylab.loglog(r_arr, ein_ds, linestyle=':')

pylab.subplot(1, 2, 2)

pylab.loglog(r_arr/nfw_lens.Mpc2deg, nfw_Sigmabar, linestyle='--')
pylab.loglog(r_arr/nfw_lens.Mpc2deg, ein_Sigmabar, linestyle=':')

pylab.show()


