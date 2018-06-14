import pylab
import numpy as np
import wl_lens_models
import lens_tools



zd = 0.3
m200 = 1e13
m200 = 1.
mstar = 1e11
reff = 1.
c200 = 5.

zs = 1.

old_lens = lens_tools.NFWdeVProfile(z=zd, m200=m200, mstar=mstar, reff=reff, c200=c200)

new_lens = wl_lens_models.NFWdeV(z=zd, m200=m200, mstar=mstar, reff=reff, c200=c200)

pnt_lens = wl_lens_models.NFWPoint(z=zd, m200=m200, mstar=mstar, c200=c200)

nr = 101
r_arr = np.logspace(-2., 0., nr) * old_lens.Mpc2deg

sources = {}
sources['r'] = np.random.rand(nr) * (r_arr[-1] - r_arr[0]) + r_arr[0]
sources['z'] = np.random.rand(nr) + 0.5 + zd

old_gamma = 0. * r_arr
new_gamma = 0. * r_arr
pnt_gamma = 0. * r_arr

old_kappa = 0. * r_arr
new_kappa = 0. * r_arr
pnt_kappa = 0. * r_arr

old_ds = 0. * r_arr
new_ds = 0. * r_arr
pnt_ds = 0. * r_arr

old_Sigmabar = 0. * r_arr
new_Sigmabar = 0. * r_arr
pnt_Sigmabar  = 0. * r_arr

for i in range(nr):

    old_gamma[i] = old_lens.gammat(r_arr[i], zs)
    new_gamma[i] = new_lens.gammat(r_arr[i], zs)
    pnt_gamma[i] = pnt_lens.gammat(r_arr[i], zs)

    old_kappa[i] = old_lens.kappa(r_arr[i], zs)
    new_kappa[i] = new_lens.kappa(r_arr[i], zs)
    pnt_kappa[i] = pnt_lens.kappa(r_arr[i], zs)

    old_ds[i] = old_lens.DeltaSigma(r_arr[i])
    new_ds[i] = new_lens.DeltaSigma(r_arr[i])
    pnt_ds[i] = pnt_lens.DeltaSigma(r_arr[i])

    old_Sigmabar[i] = old_lens.Sigmabar(r_arr[i])
    new_Sigmabar[i] = new_lens.Sigmabar(r_arr[i])
    pnt_Sigmabar[i] = pnt_lens.Sigmabar(r_arr[i])

"""
pylab.subplot(1, 2, 1)
pylab.loglog(r_arr, old_gamma)
pylab.loglog(r_arr, new_gamma, linestyle='--')
pylab.loglog(r_arr, pnt_gamma, linestyle=':')

pylab.subplot(1, 2, 2)
pylab.loglog(r_arr, old_kappa)
pylab.loglog(r_arr, new_kappa, linestyle='--')
pylab.loglog(r_arr, pnt_kappa, linestyle=':')

pylab.show()

old_lens.sources = sources
old_lens.get_source_scr()

new_lens.sources = sources
new_lens.get_source_scr()

pnt_lens.sources = sources
pnt_lens.get_source_scr()

old_lens.get_source_kappa()
old_lens.get_source_gammat()

new_lens.get_source_kappa()
new_lens.get_source_gammat()

pnt_lens.get_source_kappa()
pnt_lens.get_source_gammat()

pylab.subplot(1, 2, 1)

pylab.scatter(old_lens.sources['gammat'], new_lens.sources['gammat'], marker='x')
pylab.scatter(old_lens.sources['gammat'], pnt_lens.sources['gammat'], marker='.')
xlim = pylab.xlim()
xs = np.linspace(xlim[0], xlim[1])
pylab.plot(xs, xs, linestyle=':', color='k')

pylab.subplot(1, 2, 2)

pylab.scatter(old_lens.sources['kappa'], new_lens.sources['kappa'], marker='x')
pylab.scatter(old_lens.sources['kappa'], pnt_lens.sources['kappa'], marker='.')
xlim = pylab.xlim()
xs = np.linspace(xlim[0], xlim[1])
pylab.plot(xs, xs, linestyle=':', color='k')

pylab.show()
"""

pylab.subplot(1, 2, 1)

pylab.loglog(r_arr, old_ds)
pylab.loglog(r_arr, new_ds, linestyle='--')
pylab.loglog(r_arr, pnt_ds, linestyle=':')

pylab.subplot(1, 2, 2)

pylab.loglog(r_arr/old_lens.Mpc2deg, old_Sigmabar)
pylab.loglog(r_arr/old_lens.Mpc2deg, new_Sigmabar, linestyle='--')
pylab.loglog(r_arr/old_lens.Mpc2deg, pnt_Sigmabar, linestyle=':')

pylab.show()


