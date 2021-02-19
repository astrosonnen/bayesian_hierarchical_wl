import numpy as np
import pylab
import wl_lens_models
from matplotlib.patches import Ellipse


# draws the shapes of lensed sources around a lens
nsource = 400
nframes = 30
fsize = 20

seedno = 0
np.random.seed(seedno)

# lens and source redshift (fixed for all)
zd = 0.3
zs = 1.0

# three lenses
m200_0 = np.array([1e13, 1e12, 1e13]) # halo mass starting values

m200_seq = np.tile(m200_0, (nframes, 1))

# first varies the mass of lens 1
m200_seq[:5, 0] = m200_0[0] * (np.arange(5) + 1)
m200_seq[5:10, 0] = np.flipud(m200_0[0] * (np.arange(5) + 1))

# then lens 2
m200_seq[10:15, 1] = m200_0[1] * (np.arange(5) + 1)
m200_seq[15:20, 1] = np.flipud(m200_0[1] * (np.arange(5) + 1))

# then lens 3
m200_seq[20:25, 2] = m200_0[2] * (np.arange(5) + 1)
m200_seq[25:, 2] = np.flipud(m200_0[2] * (np.arange(5) + 1))

# lens model: NFW halo + point mass at the center
lens1 = wl_lens_models.NFWPoint(z=zd, m200=m200_0[0], mstar=1e11, c200=5.)
lens2 = wl_lens_models.NFWPoint(z=zd, m200=m200_0[1], mstar=1e11, c200=5.)
lens3 = wl_lens_models.NFWPoint(z=zd, m200=m200_0[2], mstar=1e11, c200=5.)
lenses = [lens1, lens2, lens3]

nlens = len(lenses)

# lens positions
xl = np.array([0., -0.1, 1.]) * lens1.Mpc2deg
yl = np.array([0., 0.2, 0.]) * lens1.Mpc2deg

# box half-size in degrees (16:9 aspect ratio to put in slide)
xmax = 1.6*lens1.Mpc2deg # 1.6 Mpc 
ymax = 0.9*lens1.Mpc2deg # 0.9 Mpc

# draws source positions uniformly in a rectangle of size 2.*xmax x 2.*ymax
x_source = 2.*xmax * (np.random.rand(nsource) - 0.5)
y_source = 2.*ymax * (np.random.rand(nsource) - 0.5)

# generates intrinsic shapes. Circular, if sigma_eps is set to 0.
sigma_eps = 0. # 0.27
eps11s = np.random.normal(0., sigma_eps, nsource)
eps12s = np.random.normal(0., sigma_eps, nsource)

for frameno in range(nframes):
    print(frameno)

    kappa = np.zeros(nsource)
    gamma1 = np.zeros(nsource)
    gamma2 = np.zeros(nsource)

    # calculates the contribution of each lens to kappa and gamma
    for n in range(nlens):
        # updates the halo mass
        lenses[n].m200 = m200_seq[frameno, n]
        lenses[n].update()

        # transforms to polar coordinates
        r_source = ((x_source-xl[n])**2 + (y_source-yl[n])**2)**0.5
        phi_source = np.arctan((y_source-yl[n])/(x_source-xl[n]))
        phi_source[x_source<0.] = phi_source[x_source<0.] + np.pi
        phi_source[phi_source<0.] += 2.*np.pi
    
        # calculates amplitude and direction of the reduced shear at each source position
        kappa += lenses[n].kappa(r_source, zs)
        gamma1 += lenses[n].gamma1(r_source, phi_source, zs)
        gamma2 += lenses[n].gamma2(r_source, phi_source, zs)
    
    gamma = (gamma1**2 + gamma2**2)**0.5
    
    gamma_pa = np.arctan(gamma2/gamma1)
    gamma_pa[gamma1<0.] = gamma_pa[gamma1<0.] + np.pi
    gamma_pa[gamma_pa<0.] += 2.*np.pi
    
    # lensing-induced ellipticity
    q = (1. - kappa - gamma)/(1. - kappa + gamma)
    crazy_q = 1. - 10.*(1.-q) # enhanced ellipticity
    crazy_q[crazy_q < 0.2] = 0.2
    
    ncirc = 101
    phi_circ = 2.*np.pi*np.linspace(0., 1., ncirc)
    ssize = 0.03*xmax
    r_circ = ssize * np.ones(ncirc)
    x_circ = r_circ * np.cos(phi_circ)
    y_circ = r_circ * np.sin(phi_circ)
    
    fig = pylab.figure(figsize=(16., 9.))
    ax = fig.add_subplot(111)#, aspect='equal')
    pylab.subplots_adjust(left=0., right=1., bottom=0., top=1.)
    
    for i in range(nsource):
    
        ell = Ellipse(xy=(x_source[i], y_source[i]), width=ssize*crazy_q[i], height=ssize/crazy_q[i], angle=np.rad2deg(0.5*gamma_pa[i]))
    
        ax.add_artist(ell)
    
    text_dx = -0.7*ssize
    text_dy = -0.4*ssize
    
    boxtext = ''
    for n in range(nlens):
        # paints lens galaxy
        ell = Ellipse(xy=(xl[n], yl[n]), width=2.*ssize, height=2.*ssize, angle=0., color='r')
        ax.add_artist(ell)
        ax.text(xl[n]+text_dx, yl[n]+text_dy, '$M_{%d}$'%(n+1), fontsize=fsize)
        boxtext += '$M_{%d} = %2.1e\,M_{\odot}$\n'%(n+1, m200_seq[frameno, n])

        # paints virial radius
        ell = Ellipse(xy=(xl[n], yl[n]), width=2.*lenses[n].r200*lenses[n].Mpc2deg, height=2.*lenses[n].r200*lenses[n].Mpc2deg, angle=0., color='k', fill=False)
        ax.add_artist(ell)
    boxtext = boxtext[:-1]
    
    # adds a legend
    props = dict(boxstyle='round', facecolor='wheat', alpha=1.)
    ax.text(-0.9*xmax, 0.7*ymax, boxtext, fontsize=fsize, bbox=props)
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-ymax, ymax)
    
    # adds a ruler
    ax.errorbar(0.8*xmax, 0.8*ymax, yerr=0., xerr=0.1*lenses[0].Mpc2deg, capsize=5, color='k')
    ax.text(0.72*xmax, 0.82*ymax, '200 kpc', fontsize=fsize)

    pylab.xticks(())
    pylab.yticks(())
    pylab.axis('off')
    pylab.savefig('figs/three_lenses_%02d.png'%frameno)
    pylab.close()


