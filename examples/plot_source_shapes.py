import numpy as np
import pylab
import wl_lens_models
from matplotlib.patches import Ellipse


# draws the shapes of lensed sources around a lens
nsource = 400

# background color
bkg_color = 'k'
pylab.rcParams['axes.facecolor'] = bkg_color

# lens color
lcolor = (1., 1., 0.2)

# source color
scolor = (0., 0.5, 1.)

# lens and source redshift (fixed for all)
zd = 0.3
zs = 1.0

minell = 0.3 # minimum allowed ellipticity

# lens model: NFW halo + point mass at the center
lens = wl_lens_models.NFWPoint(z=zd, m200=1e13, mstar=1e11, c200=5.)

# box half-size in degrees (16:9 aspect ratio to put in slide)
xmax = 0.8*lens.Mpc2deg # 1.6 Mpc 
ymax = 0.45*lens.Mpc2deg # 0.9 Mpc

np.random.seed(0)

# draws source positions uniformly in a rectangle of size 2.*xmax x 2.*ymax
x_source = 2.*xmax * (np.random.rand(nsource) - 0.5)
y_source = 2.*ymax * (np.random.rand(nsource) - 0.5)

# transforms to polar coordinates
r_source = (x_source**2 + y_source**2)**0.5
phi_source = np.arctan(y_source/x_source)
phi_source[x_source<0.] = phi_source[x_source<0.] + np.pi
phi_source[phi_source<0.] += 2.*np.pi

# generates intrinsic shapes. Circular, if sigma_eps is set to 0.
sigma_eps = 0.27
eps11s = np.random.normal(0., sigma_eps, nsource)
eps12s = np.random.normal(0., sigma_eps, nsource)

# calculates amplitude and direction of the reduced shear at each source position
kappa = lens.kappa(r_source, zs)
gamma1 = lens.gamma1(r_source, phi_source, zs)
gamma2 = lens.gamma2(r_source, phi_source, zs)
gamma = (gamma1**2 + gamma2**2)**0.5

gamma_pa = np.arctan(gamma2/gamma1)
gamma_pa[gamma1<0.] = gamma_pa[gamma1<0.] + np.pi
gamma_pa[gamma_pa<0.] += 2.*np.pi

# lensing-induced ellipticity
q = (1. - kappa - gamma)/(1. - kappa + gamma)
crazy_q = 1. - 10.*(1.-q) # enhanced ellipticity
crazy_q[crazy_q < minell] = minell # avoids crazy large elongations

compell = eps11s + 1j*eps12s # intrinsic complex ellipticity

gammac = gamma1 + 1j*gamma2

gcompl = gammac/(1. - kappa)

compell = (compell + gcompl) / (1. + np.conj(compell)*np.conj(gcompl)) # observed complex ellipticity

et = np.absolute(compell)
shape_pa = np.angle(compell)

shape_q = (1. - et)/(1 + et) # is this ok???
shape_q[shape_q < minell] = minell # avoids crazy large elongations

#tanphi = gamma1/gamma2 - ((gamma1/gamma2)**2 + 1.)**0.5
#pa = np.arctan(tanphi)

ncirc = 101
phi_circ = 2.*np.pi*np.linspace(0., 1., ncirc)
ssize = 0.03*xmax
r_circ = ssize * np.ones(ncirc)
x_circ = r_circ * np.cos(phi_circ)
y_circ = r_circ * np.sin(phi_circ)

fig = pylab.figure(figsize=(16., 9.))
ax = fig.add_subplot(111)
pylab.subplots_adjust(left=0., right=1., bottom=0., top=1.)

for i in range(nsource):

    ell = Ellipse(xy=(x_source[i], y_source[i]), width=ssize*q[i], height=ssize/q[i], angle=np.rad2deg(0.5*gamma_pa[i]), color=scolor)

    ax.add_artist(ell)

ell = Ellipse(xy=(0., 0.), width=2.*ssize, height=2.*ssize, angle=0., color=lcolor)
ax.add_artist(ell)

# paints virial radius
ell = Ellipse(xy=(0., 0.), width=2.*lens.r200*lens.Mpc2deg, height=2.*lens.r200*lens.Mpc2deg, angle=0., color=lcolor, fill=False)
ax.add_artist(ell)

ax.set_xlim(-xmax, xmax)
ax.set_ylim(-ymax, ymax)

pylab.xticks(())
pylab.yticks(())
#pylab.axis('off')
ax.set_facecolor(bkg_color)
pylab.savefig('shapes_original_circular_plot.png')
pylab.show()

fig = pylab.figure(figsize=(16., 9.))
ax = fig.add_subplot(111)
pylab.subplots_adjust(left=0., right=1., bottom=0., top=1.)

for i in range(nsource):

    ell = Ellipse(xy=(x_source[i], y_source[i]), width=ssize*crazy_q[i], height=ssize/crazy_q[i], angle=np.rad2deg(0.5*gamma_pa[i]), color=scolor)

    ax.add_artist(ell)

ell = Ellipse(xy=(0., 0.), width=2.*ssize, height=2.*ssize, angle=0., color=lcolor)
ax.add_artist(ell)

# paints virial radius
ell = Ellipse(xy=(0., 0.), width=2.*lens.r200*lens.Mpc2deg, height=2.*lens.r200*lens.Mpc2deg, angle=0., color=lcolor, fill=False)
ax.add_artist(ell)

ax.set_xlim(-xmax, xmax)
ax.set_ylim(-ymax, ymax)

pylab.xticks(())
pylab.yticks(())
#pylab.axis('off')
ax.set_facecolor(bkg_color)
pylab.savefig('shapes_enhanced_circular_plot.png')
pylab.show()


fig = pylab.figure(figsize=(16., 9.))
pylab.subplots_adjust(left=0., right=1., bottom=0., top=1.)
ax = fig.add_subplot(111)

for i in range(nsource):

    ell = Ellipse(xy=(x_source[i], y_source[i]), width=ssize*shape_q[i], height=ssize/shape_q[i], angle=np.rad2deg(0.5*shape_pa[i]), color=scolor)

    ax.add_artist(ell)

ell = Ellipse(xy=(0., 0.), width=2.*ssize, height=2.*ssize, angle=0., color=lcolor)
ax.add_artist(ell)

# paints virial radius
ell = Ellipse(xy=(0., 0.), width=2.*lens.r200*lens.Mpc2deg, height=2.*lens.r200*lens.Mpc2deg, angle=0., color=lcolor, fill=False)
ax.add_artist(ell)

ax.set_xlim(-xmax, xmax)
ax.set_ylim(-ymax, ymax)

pylab.xticks(())
pylab.yticks(())
#pylab.axis('off')
ax.set_facecolor(bkg_color)
pylab.savefig('shapes_original_plot.png')
pylab.show()

