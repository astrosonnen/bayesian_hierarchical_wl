import numpy as np
import wl_cosmology
from wl_cosmology import c, G, Mpc, M_Sun
from wl_profiles import nfw, deVaucouleurs, gnfw, einasto, sersic
from scipy.optimize import brentq
from math import pi


class NFWPoint:

    def __init__(self, z=0.3, m200=1e13, c200=5., mstar=1e11, ra=0., dec=0., sources=None, \
                 cosmo=wl_cosmology.default_cosmo):

        self.z = z
        self.cosmo = cosmo
        self.m200 = m200 # halo mass in M_Sun units
        self.mstar = mstar # central point mass in M_Sun units
        self.rhocrit = wl_cosmology.rhoc(self.z, cosmo=self.cosmo)
        self.r200 = (self.m200*3./200./(4.*pi)/self.rhocrit)**(1/3.) #r200 in Mpc
        self.c200 = c200
        self.rs = self.r200/self.c200
        self.angD = wl_cosmology.Dang(self.z, cosmo=self.cosmo)
        self.Mpc2deg = np.rad2deg(1./self.angD)
        self.rs_ang = self.rs*self.Mpc2deg
        self.S_s = self.m200/(np.log(1. + self.c200) - self.c200/(1. + self.c200))/(4.*pi*self.rs**2)
        self.sources = sources

        if ra is not None:
            self.ra = ra
        if dec is not None:
            self.dec = dec

    def update(self):
        self.r200 = (self.m200*3./200./(4.*pi)/self.rhocrit)**(1/3.)
        self.rs = self.r200/self.c200
        self.rs_ang = self.rs*self.Mpc2deg
        self.S_s = self.m200/(np.log(1. + self.c200) - self.c200/(1. + self.c200))/(4.*pi*self.rs**2)

    def S_cr(self, z):
        Ds = wl_cosmology.Dang(z, cosmo=self.cosmo)
        Dds = wl_cosmology.Dang(z, self.z, cosmo=self.cosmo)
        return c**2/(4.*pi*G)*Ds/Dds/self.angD*Mpc/M_Sun

    def get_source_scr(self):
        nsource = len(self.sources['z'])
        s_crs = np.zeros(nsource)
        for i in range(nsource):
            if self.sources['z'][i] > self.z:
                s_crs[i] = self.S_cr(self.sources['z'][i])
        self.sources['s_cr'] = s_crs

    def get_source_polarcoords(self):

        self.sources['r'] = ((self.ra - self.sources['ra'])**2*np.cos(np.deg2rad(self.dec))**2 + (self.dec - self.sources['dec'])**2)**0.5

        xh =  - (self.sources['ra'] - self.ra)*np.cos(np.deg2rad(self.dec))
        yh = self.sources['dec'] - self.dec

        phih = np.arctan(yh/xh)

        phih[xh<0.] = phih[xh<0.] + np.pi
        phih[phih<0.] += 2.*np.pi

        self.sources['phi'] = phih

    def get_source_et(self):

        cosphi = np.cos(self.sources['phi'])
        sinphi = np.sin(self.sources['phi'])

        sin2phi = 2.*cosphi*sinphi
        cos2phi = cosphi**2 - sinphi**2

        self.sources['et'] = self.sources['e1']*cos2phi + self.sources['e2']*sin2phi

    def get_source_gammat(self):
        self.sources['gammat'] = (2.*self.S_s*(2.*nfw.gfunc(self.sources['r']/self.rs_ang)/(self.sources['r']/self.rs_ang)**2 + \
                             - nfw.Ffunc(self.sources['r']/self.rs_ang)) + \
                self.mstar/(self.sources['r']/self.Mpc2deg)**2/pi) / self.sources['s_cr']

    def get_source_kappa(self):
        self.sources['kappa'] = 2.*self.S_s*nfw.Ffunc(self.sources['r']/self.rs_ang) / self.sources['s_cr']

    def gammat(self, theta, z):
        return (2.*self.S_s*(2.*nfw.gfunc(theta/self.rs_ang)/(theta/self.rs_ang)**2 - nfw.Ffunc(theta/self.rs_ang)) + \
                self.mstar/(theta/self.Mpc2deg)**2/pi) / self.S_cr(z)

    def gamma1(self, r, phi, z):
        return np.cos(2.*phi)*self.gammat(r, z)

    def gamma2(self, r, phi, z):
        return np.sin(2.*phi)*self.gammat(r, z)

    def kappa(self, theta, z):
        return 2.*self.S_s*nfw.Ffunc(theta/self.rs_ang) / self.S_cr(z)

    def m(self, theta, z):
        return (4.*self.S_s*self.rs_ang**2*nfw.gfunc(theta/self.rs_ang) + \
                self.mstar/pi/self.Mpc2deg**2) / self.S_cr(z)

    def alpha(self, theta, z):
        return (4.*self.S_s*theta/(theta/self.rs_ang)**2*nfw.gfunc(theta/self.rs_ang) + \
               self.mstar/pi*theta/(1./self.Mpc2deg)**2) / self.S_cr(z)

    def mu(self, theta, z):
        return ((1 - self.kappa(theta, z))**2 - self.gammat(theta, z)**2)**(-1)

    def rein(self, z, xtol=1e-6, xmin=1e-6, xmax=1.):
        bfunc = lambda theta: theta - self.alpha(theta, z)
        if bfunc(xmin)*bfunc(xmax) > 0:
            return 0.
        else:
            return brentq(bfunc, xmin, xmax, xtol=xtol)

    def gcompl(self, r, phi, z):
        return (1. - self.kappa(r, z))**(-1)*(self.gamma1(r, phi, z) + 1j*self.gamma2(r, phi, z))

    def gcomplstar(self, r, phi, z):
        return (1. - self.kappa(r, z))**(-1)*(self.gamma1(r, phi, z) - 1j*self.gamma2(r, phi, z))

    def source_gcompl(self):
        self.get_source_gammat()
        self.get_source_kappa()
        return (1. - self.sources['kappa'])**(-1) * self.sources['gammat'] * \
               (np.cos(2.*self.sources['phi']) + 1j*np.sin(2.*self.sources['phi']))

    def source_gamma(self):
        self.get_source_gammat()
        return self.sources['gammat'] * (np.cos(2.*self.sources['phi']) + 1j*np.sin(2.*self.sources['phi']))

    def get_image(self, y, z, rmin=None, rmax=10.):

        if rmin is None:
            rmin = self.rein(z)

        xfunc = lambda theta: theta - self.alpha(theta, z) - y

        return brentq(xfunc, rmin, rmax)

    def Sigma(self, theta): # surface mass density in M_Sun/pc^2
        return (2.*self.S_s*nfw.Ffunc(theta/self.rs_ang))/1e12

    def Sigmabar(self, theta):
        return (4.*self.S_s*nfw.gfunc(theta/self.rs_ang)/(theta/self.rs_ang)**2 + \
                self.mstar/(theta/self.Mpc2deg)**2/pi) /1e12

    def DeltaSigma(self, theta):
        return (self.Sigmabar(theta) - self.Sigma(theta))#/cosmo['h']


class GNFWPoint:

    def __init__(self, z=0.3, m200=1e13, c200=5., gamma=1., mstar=1e11, ra=0., dec=0., sources=None, \
                 cosmo=wl_cosmology.default_cosmo):

        self.z = z
        self.cosmo = cosmo
        self.m200 = m200 # halo mass in M_Sun units
        self.mstar = mstar # central point mass in M_Sun units
        self.rhocrit = wl_cosmology.rhoc(self.z, cosmo=self.cosmo)
        self.r200 = (self.m200*3./200./(4.*pi)/self.rhocrit)**(1/3.) #r200 in Mpc
        self.c200 = c200
        self.rs = self.r200/self.c200
        self.gamma = gamma
        self.angD = wl_cosmology.Dang(self.z, cosmo=self.cosmo)
        self.Mpc2deg = np.rad2deg(1./self.angD)
        self.rs_ang = self.rs*self.Mpc2deg
        self.halo_norm = self.m200/gnfw.fast_M3d(self.r200, self.rs, self.gamma)
        self.sources = sources

        if ra is not None:
            self.ra = ra
        if dec is not None:
            self.dec = dec

    def update(self):
        self.r200 = (self.m200*3./200./(4.*pi)/self.rhocrit)**(1/3.)
        self.rs = self.r200/self.c200
        self.rs_ang = self.rs*self.Mpc2deg
        self.halo_norm = self.m200/gnfw.fast_M3d(self.r200, self.rs, self.gamma)

    def S_cr(self, z):
        Ds = wl_cosmology.Dang(z, cosmo=self.cosmo)
        Dds = wl_cosmology.Dang(z, self.z, cosmo=self.cosmo)
        return c**2/(4.*pi*G)*Ds/Dds/self.angD*Mpc/M_Sun

    def get_source_scr(self):
        nsource = len(self.sources['z'])
        s_crs = np.zeros(nsource)
        for i in range(nsource):
            if self.sources['z'][i] > self.z:
                s_crs[i] = self.S_cr(self.sources['z'][i])
        self.sources['s_cr'] = s_crs

    def get_source_polarcoords(self):

        self.sources['r'] = ((self.ra - self.sources['ra'])**2*np.cos(np.deg2rad(self.dec))**2 + (self.dec - self.sources['dec'])**2)**0.5

        xh =  - (self.sources['ra'] - self.ra)*np.cos(np.deg2rad(self.dec))
        yh = self.sources['dec'] - self.dec

        phih = np.arctan(yh/xh)

        phih[xh<0.] = phih[xh<0.] + np.pi
        phih[phih<0.] += 2.*np.pi

        self.sources['phi'] = phih

    def get_source_et(self):

        cosphi = np.cos(self.sources['phi'])
        sinphi = np.sin(self.sources['phi'])

        sin2phi = 2.*cosphi*sinphi
        cos2phi = cosphi**2 - sinphi**2

        self.sources['et'] = self.sources['e1']*cos2phi + self.sources['e2']*sin2phi

    def get_source_gammat(self):
        self.sources['gammat'] = (self.halo_norm*(gnfw.fast_M2d(self.sources['r'], self.rs_ang, self.gamma)/pi/(self.sources['r']/self.Mpc2deg)**2 - gnfw.fast_Sigma(self.sources['r']/self.Mpc2deg, self.rs, self.gamma)) + \
                self.mstar/(self.sources['r']/self.Mpc2deg)**2/pi) / self.sources['s_cr']

    def get_source_kappa(self):
        self.sources['kappa'] = self.halo_norm*gnfw.fast_Sigma(self.sources['r']/self.Mpc2deg, self.rs, self.gamma) / self.sources['s_cr']

    def gammat(self, theta, z):
        return (self.halo_norm*(gnfw.fast_M2d(theta, self.rs_ang, self.gamma)/pi/(theta/self.Mpc2deg)**2 - gnfw.fast_Sigma(theta/self.Mpc2deg, self.rs, self.gamma)) + \
                self.mstar/(theta/self.Mpc2deg)**2/pi) / self.S_cr(z)

    def gamma1(self, r, phi, z):
        return np.cos(2.*phi)*self.gammat(r, z)

    def gamma2(self, r, phi, z):
        return np.sin(2.*phi)*self.gammat(r, z)

    def kappa(self, theta, z):
        return self.halo_norm*gnfw.fast_Sigma(theta/self.Mpc2deg, self.rs, self.gamma) / self.S_cr(z)

    def m(self, theta, z):
        return (self.halo_norm*gnfw.fast_M2d(theta, self.rs_ang, self.gamma) + self.mstar)/ (pi * self.S_cr(z) / self.Mpc2deg**2) 

    def alpha(self, theta, z):
        return (self.halo_norm/theta*gnfw.fast_M2d(theta/self.rs_ang) + \
               self.mstar/pi*theta/(1./self.Mpc2deg)**2) / self.S_cr(z)

    def mu(self, theta, z):
        return ((1 - self.kappa(theta, z))**2 - self.gammat(theta, z)**2)**(-1)

    def rein(self, z, xtol=1e-6, xmin=1e-6, xmax=1.):
        bfunc = lambda theta: theta - self.alpha(theta, z)
        if bfunc(xmin)*bfunc(xmax) > 0:
            return 0.
        else:
            return brentq(bfunc, xmin, xmax, xtol=xtol)

    def gcompl(self, r, phi, z):
        return (1. - self.kappa(r, z))**(-1)*(self.gamma1(r, phi, z) + 1j*self.gamma2(r, phi, z))

    def gcomplstar(self, r, phi, z):
        return (1. - self.kappa(r, z))**(-1)*(self.gamma1(r, phi, z) - 1j*self.gamma2(r, phi, z))

    def source_gcompl(self):
        self.get_source_gammat()
        self.get_source_kappa()
        return (1. - self.sources['kappa'])**(-1) * self.sources['gammat'] * \
               (np.cos(2.*self.sources['phi']) + 1j*np.sin(2.*self.sources['phi']))

    def source_gamma(self):
        self.get_source_gammat()
        return self.sources['gammat'] * (np.cos(2.*self.sources['phi']) + 1j*np.sin(2.*self.sources['phi']))

    def get_image(self, y, z, rmin=None, rmax=10.):

        if rmin is None:
            rmin = self.rein(z)

        xfunc = lambda theta: theta - self.alpha(theta, z) - y

        return brentq(xfunc, rmin, rmax)

    def Sigma(self, theta): # surface mass density in M_Sun/pc^2
        return (self.halo_norm*gnfw.fast_Sigma(theta/self.Mpc2deg, self.rs, self.gamma))/1e12

    def Sigmabar(self, theta):
        return (self.halo_norm*gnfw.fast_M2d(theta, self.rs_ang, self.gamma)/(theta/self.Mpc2deg)**2/pi + \
                self.mstar/(theta/self.Mpc2deg)**2/pi) /1e12

    def DeltaSigma(self, theta):
        return (self.Sigmabar(theta) - self.Sigma(theta))


class NFWdeV:

    def __init__(self, z=0.3, m200=1e13, c200=5., mstar=1e11, reff=5., ra=0., dec=0., sources=None, \
                 cosmo=wl_cosmology.default_cosmo):


        self.z = z
        self.cosmo = cosmo
        self.m200 = m200 # halo mass in M_Sun units
        self.mstar = mstar # stellar mass in M_Sun units
        self.rhocrit = wl_cosmology.rhoc(self.z, cosmo=self.cosmo)
        self.r200 = (self.m200*3./200./(4.*pi)/self.rhocrit)**(1/3.) #r200 in Mpc
        self.c200 = c200
        self.rs = self.r200/self.c200
        self.reff = reff # effective radius in kpc
        self.angD = wl_cosmology.Dang(self.z, cosmo=self.cosmo)
        self.Mpc2deg = np.rad2deg(1./self.angD)
        self.rs_ang = self.rs*self.Mpc2deg
        self.reff_ang = self.reff*self.Mpc2deg/1000.
        self.S_s = self.m200/(np.log(1. + self.c200) - self.c200/(1. + self.c200))/(4.*pi*self.rs**2)
        self.S_bulge = self.mstar/self.reff**2*1e6
        self.sources = sources

        if ra is not None:
            self.ra = ra
        if dec is not None:
            self.dec = dec

    def update(self):
        self.r200 = (self.m200*3./200./(4.*pi)/self.rhocrit)**(1/3.)
        self.rs = self.r200/self.c200
        self.rs_ang = self.rs*self.Mpc2deg
        self.reff_ang = self.reff*self.Mpc2deg/1000.
        self.S_s = self.m200/(np.log(1. + self.c200) - self.c200/(1. + self.c200))/(4.*pi*self.rs**2)
        self.S_bulge = self.mstar/self.reff**2*1e6

    def S_cr(self, z):
        Ds = wl_cosmology.Dang(z, cosmo=self.cosmo)
        Dds = wl_cosmology.Dang(z, self.z, cosmo=self.cosmo)
        return c**2/(4.*pi*G)*Ds/Dds/self.angD*Mpc/M_Sun

    def get_source_scr(self):
        nsource = len(self.sources['z'])
        s_crs = np.zeros(nsource)
        for i in range(nsource):
            if self.sources['z'][i] > self.z:
                s_crs[i] = self.S_cr(self.sources['z'][i])
        self.sources['s_cr'] = s_crs

    def get_source_polarcoords(self):

        self.sources['r'] = ((self.ra - self.sources['ra'])**2*np.cos(np.deg2rad(self.dec))**2 + (self.dec - self.sources['dec'])**2)**0.5

        xh =  - (self.sources['ra'] - self.ra)*np.cos(np.deg2rad(self.dec))
        yh = self.sources['dec'] - self.dec

        phih = np.arctan(yh/xh)

        phih[xh<0.] = phih[xh<0.] + np.pi
        phih[phih<0.] += 2.*np.pi

        self.sources['phi'] = phih

    def get_source_et(self):

        cosphi = np.cos(self.sources['phi'])
        sinphi = np.sin(self.sources['phi'])

        sin2phi = 2.*cosphi*sinphi
        cos2phi = cosphi**2 - sinphi**2

        self.sources['et'] = self.sources['e1']*cos2phi + self.sources['e2']*sin2phi

    def get_source_gammat(self):
        self.sources['gammat'] = (2.*self.S_s*(2.*nfw.gfunc(self.sources['r']/self.rs_ang)/(self.sources['r']/self.rs_ang)**2 + \
                             - nfw.Ffunc(self.sources['r']/self.rs_ang)) + \
                self.S_bulge*(deVaucouleurs.fast_M2d(self.sources['r']/self.reff_ang)/(self.sources['r']/self.reff_ang)**2/pi - \
                              deVaucouleurs.Sigma(self.sources['r']/self.reff_ang, 1.))) / self.sources['s_cr']

    def get_source_kappa(self):
        self.sources['kappa'] = (2.*self.S_s*nfw.Ffunc(self.sources['r']/self.rs_ang) + \
                self.S_bulge*deVaucouleurs.Sigma(self.sources['r']/self.reff_ang, 1.)) / self.sources['s_cr']

    def gammat(self, theta, z):
        return (2.*self.S_s*(2.*nfw.gfunc(theta/self.rs_ang)/(theta/self.rs_ang)**2 - nfw.Ffunc(theta/self.rs_ang)) + \
                self.S_bulge*(deVaucouleurs.fast_M2d(theta/self.reff_ang)/(theta/self.reff_ang)**2/pi - \
                              deVaucouleurs.Sigma(theta/self.reff_ang, 1.))) / self.S_cr(z)

    def gamma1(self, r, phi, z):
        return np.cos(2.*phi)*self.gammat(r, z)

    def gamma2(self, r, phi, z):
        return np.sin(2.*phi)*self.gammat(r, z)

    def kappa(self, theta, z):
        return (2.*self.S_s*nfw.Ffunc(theta/self.rs_ang) + self.S_bulge*deVaucouleurs.Sigma(theta/self.reff_ang, 1.)) /self.S_cr(z)

    def m(self, theta, z):
        return (4.*self.S_s*self.rs_ang**2*nfw.gfunc(theta/self.rs_ang) + \
                self.S_bulge/pi*self.reff_ang**2*deVaucouleurs.fast_M2d(theta/self.reff_ang)) / self.S_cr(z)

    def alpha(self, theta, z):
        return (4.*self.S_s*theta/(theta/self.rs_ang)**2*nfw.gfunc(theta/self.rs_ang) + \
               self.S_bulge/pi*theta/(theta/self.reff_ang)**2*deVaucouleurs.fast_M2d(theta/self.reff_ang)) / self.S_cr(z)

    def mu(self, theta, z):
        return ((1 - self.kappa(theta, z))**2 - self.gammat(theta, z)**2)**(-1)

    def rein(self, z, xtol=1e-6, xmin=1e-6, xmax=1.):
        bfunc = lambda theta: theta - self.alpha(theta, z)
        if bfunc(xmin)*bfunc(xmax) > 0:
            return 0.
        else:
            return brentq(bfunc, xmin, xmax, xtol=xtol)

    def gcompl(self, r, phi, z):
        return (1. - self.kappa(r, z))**(-1)*(self.gamma1(r, phi, z) + 1j*self.gamma2(r, phi, z))

    def gcomplstar(self, r, phi, z):
        return (1. - self.kappa(r, z))**(-1)*(self.gamma1(r, phi, z) - 1j*self.gamma2(r, phi, z))

    def source_gcompl(self):
        self.get_source_gammat()
        self.get_source_kappa()
        return (1. - self.sources['kappa'])**(-1) * self.sources['gammat'] * \
               (np.cos(2.*self.sources['phi']) + 1j*np.sin(2.*self.sources['phi']))

    def source_gamma(self):
        self.get_source_gammat()
        return self.sources['gammat'] * (np.cos(2.*self.sources['phi']) + 1j*np.sin(2.*self.sources['phi']))

    def get_image(self, y, z, rmin=None, rmax=10.):

        if rmin is None:
            rmin = self.rein(z)

        xfunc = lambda theta: theta - self.alpha(theta, z) - y

        return brentq(xfunc, rmin, rmax)

    def Sigma(self, theta): # surface mass density in M_Sun/pc^2
        return (2.*self.S_s*nfw.Ffunc(theta/self.rs_ang) + self.S_bulge*deVaucouleurs.Sigma(theta/self.reff_ang, 1.))/1e12

    def Sigmabar(self, theta):
        return (4.*self.S_s*nfw.gfunc(theta/self.rs_ang)/(theta/self.rs_ang)**2 + \
                self.S_bulge*(deVaucouleurs.fast_M2d(theta/self.reff_ang)/(theta/self.reff_ang)**2/pi)) /1e12

    def DeltaSigma(self, theta):
        return (self.Sigmabar(theta) - self.Sigma(theta))#/cosmo['h']

class NFWSersic:

    def __init__(self, z=0.3, m200=1e13, c200=5., mstar=1e11, reff=5., nser=4., ra=0., dec=0., sources=None, \
                 cosmo=wl_cosmology.default_cosmo):


        self.z = z
        self.cosmo = cosmo
        self.m200 = m200 # halo mass in M_Sun units
        self.mstar = mstar # stellar mass in M_Sun units
        self.rhocrit = wl_cosmology.rhoc(self.z, cosmo=self.cosmo)
        self.r200 = (self.m200*3./200./(4.*pi)/self.rhocrit)**(1/3.) #r200 in Mpc
        self.c200 = c200
        self.rs = self.r200/self.c200
        self.reff = reff # effective radius in kpc
        self.nser = nser # Sersic index
        self.angD = wl_cosmology.Dang(self.z, cosmo=self.cosmo)
        self.Mpc2deg = np.rad2deg(1./self.angD)
        self.rs_ang = self.rs*self.Mpc2deg
        self.reff_ang = self.reff*self.Mpc2deg/1000.
        self.S_s = self.m200/(np.log(1. + self.c200) - self.c200/(1. + self.c200))/(4.*pi*self.rs**2)
        self.S_bulge = self.mstar/self.reff**2*1e6
        self.sources = sources

        if ra is not None:
            self.ra = ra
        if dec is not None:
            self.dec = dec

    def update(self):
        self.r200 = (self.m200*3./200./(4.*pi)/self.rhocrit)**(1/3.)
        self.rs = self.r200/self.c200
        self.rs_ang = self.rs*self.Mpc2deg
        self.reff_ang = self.reff*self.Mpc2deg/1000.
        self.S_s = self.m200/(np.log(1. + self.c200) - self.c200/(1. + self.c200))/(4.*pi*self.rs**2)
        self.S_bulge = self.mstar/self.reff**2*1e6

    def S_cr(self, z):
        Ds = wl_cosmology.Dang(z, cosmo=self.cosmo)
        Dds = wl_cosmology.Dang(z, self.z, cosmo=self.cosmo)
        return c**2/(4.*pi*G)*Ds/Dds/self.angD*Mpc/M_Sun

    def get_source_scr(self):
        nsource = len(self.sources['z'])
        s_crs = np.zeros(nsource)
        for i in range(nsource):
            if self.sources['z'][i] > self.z:
                s_crs[i] = self.S_cr(self.sources['z'][i])
        self.sources['s_cr'] = s_crs

    def get_source_polarcoords(self):

        self.sources['r'] = ((self.ra - self.sources['ra'])**2*np.cos(np.deg2rad(self.dec))**2 + (self.dec - self.sources['dec'])**2)**0.5

        xh =  - (self.sources['ra'] - self.ra)*np.cos(np.deg2rad(self.dec))
        yh = self.sources['dec'] - self.dec

        phih = np.arctan(yh/xh)

        phih[xh<0.] = phih[xh<0.] + np.pi
        phih[phih<0.] += 2.*np.pi

        self.sources['phi'] = phih

    def get_source_et(self):

        cosphi = np.cos(self.sources['phi'])
        sinphi = np.sin(self.sources['phi'])

        sin2phi = 2.*cosphi*sinphi
        cos2phi = cosphi**2 - sinphi**2

        self.sources['et'] = self.sources['e1']*cos2phi + self.sources['e2']*sin2phi

    def get_source_gammat(self):
        self.sources['gammat'] = (2.*self.S_s*(2.*nfw.gfunc(self.sources['r']/self.rs_ang)/(self.sources['r']/self.rs_ang)**2 + \
                             - nfw.Ffunc(self.sources['r']/self.rs_ang)) + \
                self.S_bulge*(sersic.fast_M2d(self.sources['r']/self.reff_ang, self.nser)/(self.sources['r']/self.reff_ang)**2/pi - \
                              sersic.Sigma(self.sources['r']/self.reff_ang, self.nser, 1.))) / self.sources['s_cr']

    def get_source_kappa(self):
        self.sources['kappa'] = (2.*self.S_s*nfw.Ffunc(self.sources['r']/self.rs_ang) + \
                self.S_bulge*sersic.Sigma(self.sources['r']/self.reff_ang, self.nser, 1.)) / self.sources['s_cr']

    def gammat(self, theta, z):
        return (2.*self.S_s*(2.*nfw.gfunc(theta/self.rs_ang)/(theta/self.rs_ang)**2 - nfw.Ffunc(theta/self.rs_ang)) + \
                self.S_bulge*(sersic.fast_M2d(theta/self.reff_ang, self.nser)/(theta/self.reff_ang)**2/pi - \
                              sersic.Sigma(theta/self.reff_ang, self.nser, 1.))) / self.S_cr(z)

    def gamma1(self, r, phi, z):
        return np.cos(2.*phi)*self.gammat(r, z)

    def gamma2(self, r, phi, z):
        return np.sin(2.*phi)*self.gammat(r, z)

    def kappa(self, theta, z):
        return (2.*self.S_s*nfw.Ffunc(theta/self.rs_ang) + self.S_bulge*sersic.Sigma(theta/self.reff_ang, self.nser, 1.)) /self.S_cr(z)

    def m(self, theta, z):
        return (4.*self.S_s*self.rs_ang**2*nfw.gfunc(theta/self.rs_ang) + \
                self.S_bulge/pi*self.reff_ang**2*sersic.fast_M2d(theta/self.reff_ang, self.nser)) / self.S_cr(z)

    def alpha(self, theta, z):
        return (4.*self.S_s*theta/(theta/self.rs_ang)**2*nfw.gfunc(theta/self.rs_ang) + \
               self.S_bulge/pi*theta/(theta/self.reff_ang)**2*sersic.fast_M2d(theta/self.reff_ang, self.nser)) / self.S_cr(z)

    def mu(self, theta, z):
        return ((1 - self.kappa(theta, z))**2 - self.gammat(theta, z)**2)**(-1)

    def rein(self, z, xtol=1e-6, xmin=1e-6, xmax=1.):
        bfunc = lambda theta: theta - self.alpha(theta, z)
        if bfunc(xmin)*bfunc(xmax) > 0:
            return 0.
        else:
            return brentq(bfunc, xmin, xmax, xtol=xtol)

    def gcompl(self, r, phi, z):
        return (1. - self.kappa(r, z))**(-1)*(self.gamma1(r, phi, z) + 1j*self.gamma2(r, phi, z))

    def gcomplstar(self, r, phi, z):
        return (1. - self.kappa(r, z))**(-1)*(self.gamma1(r, phi, z) - 1j*self.gamma2(r, phi, z))

    def source_gcompl(self):
        self.get_source_gammat()
        self.get_source_kappa()
        return (1. - self.sources['kappa'])**(-1) * self.sources['gammat'] * \
               (np.cos(2.*self.sources['phi']) + 1j*np.sin(2.*self.sources['phi']))

    def source_gamma(self):
        self.get_source_gammat()
        return self.sources['gammat'] * (np.cos(2.*self.sources['phi']) + 1j*np.sin(2.*self.sources['phi']))

    def get_image(self, y, z, rmin=None, rmax=10.):

        if rmin is None:
            rmin = self.rein(z)

        xfunc = lambda theta: theta - self.alpha(theta, z) - y

        return brentq(xfunc, rmin, rmax)

    def Sigma(self, theta): # surface mass density in M_Sun/pc^2
        return (2.*self.S_s*nfw.Ffunc(theta/self.rs_ang) + self.S_bulge*sersic.Sigma(theta/self.reff_ang, self.nser, 1.))/1e12

    def Sigmabar(self, theta):
        return (4.*self.S_s*nfw.gfunc(theta/self.rs_ang)/(theta/self.rs_ang)**2 + \
                self.S_bulge*(sersic.fast_M2d(theta/self.reff_ang, self.nser)/(theta/self.reff_ang)**2/pi)) /1e12

    def DeltaSigma(self, theta):
        return (self.Sigmabar(theta) - self.Sigma(theta))#/cosmo['h']

class EinastoPoint:

    def __init__(self, z=0.3, m200=1e13, c200=5., alpha=1., mstar=1e11, ra=0., dec=0., sources=None, \
                 cosmo=wl_cosmology.default_cosmo):

        self.z = z
        self.cosmo = cosmo
        self.m200 = m200 # halo mass in M_Sun units
        self.mstar = mstar # central point mass in M_Sun units
        self.rhocrit = wl_cosmology.rhoc(self.z, cosmo=self.cosmo)
        self.r200 = (self.m200*3./200./(4.*pi)/self.rhocrit)**(1/3.) #r200 in Mpc
        self.c200 = c200
        self.r2 = self.r200/self.c200
        self.alpha = alpha
        self.angD = wl_cosmology.Dang(self.z, cosmo=self.cosmo)
        self.Mpc2deg = np.rad2deg(1./self.angD)
        self.r2_ang = self.r2*self.Mpc2deg
        self.halo_norm = self.m200/einasto.M3d(self.r200, self.r2, self.alpha)
        self.sources = sources

        if ra is not None:
            self.ra = ra
        if dec is not None:
            self.dec = dec

    def update(self):
        self.r200 = (self.m200*3./200./(4.*pi)/self.rhocrit)**(1/3.)
        self.r2 = self.r200/self.c200
        self.r2_ang = self.r2*self.Mpc2deg
        self.halo_norm = self.m200/einasto.M3d(self.r200, self.r2, self.alpha)

    def S_cr(self, z):
        Ds = wl_cosmology.Dang(z, cosmo=self.cosmo)
        Dds = wl_cosmology.Dang(z, self.z, cosmo=self.cosmo)
        return c**2/(4.*pi*G)*Ds/Dds/self.angD*Mpc/M_Sun

    def get_source_scr(self):
        nsource = len(self.sources['z'])
        s_crs = np.zeros(nsource)
        for i in range(nsource):
            if self.sources['z'][i] > self.z:
                s_crs[i] = self.S_cr(self.sources['z'][i])
        self.sources['s_cr'] = s_crs

    def get_source_polarcoords(self):

        self.sources['r'] = ((self.ra - self.sources['ra'])**2*np.cos(np.deg2rad(self.dec))**2 + (self.dec - self.sources['dec'])**2)**0.5

        xh =  - (self.sources['ra'] - self.ra)*np.cos(np.deg2rad(self.dec))
        yh = self.sources['dec'] - self.dec

        phih = np.arctan(yh/xh)

        phih[xh<0.] = phih[xh<0.] + np.pi
        phih[phih<0.] += 2.*np.pi

        self.sources['phi'] = phih

    def get_source_et(self):

        cosphi = np.cos(self.sources['phi'])
        sinphi = np.sin(self.sources['phi'])

        sin2phi = 2.*cosphi*sinphi
        cos2phi = cosphi**2 - sinphi**2

        self.sources['et'] = self.sources['e1']*cos2phi + self.sources['e2']*sin2phi

    def get_source_gammat(self):
        self.sources['gammat'] = (self.halo_norm*(einasto.fast_M2d(self.sources['r'], self.r2_ang, self.alpha)/pi/(self.sources['r']/self.Mpc2deg)**2 - einasto.fast_Sigma(self.sources['r']/self.Mpc2deg, self.r2, self.alpha)) + \
                self.mstar/(self.sources['r']/self.Mpc2deg)**2/pi) / self.sources['s_cr']

    def get_source_kappa(self):
        self.sources['kappa'] = self.halo_norm*einasto.fast_Sigma(self.sources['r']/self.Mpc2deg, self.r2, self.alpha) / self.sources['s_cr']

    def gammat(self, theta, z):
        return (self.halo_norm*(einasto.fast_M2d(theta, self.r2_ang, self.alpha)/pi/(theta/self.Mpc2deg)**2 - einasto.fast_Sigma(theta/self.Mpc2deg, self.r2, self.alpha)) + \
                self.mstar/(theta/self.Mpc2deg)**2/pi) / self.S_cr(z)

    def gamma1(self, r, phi, z):
        return np.cos(2.*phi)*self.gammat(r, z)

    def gamma2(self, r, phi, z):
        return np.sin(2.*phi)*self.gammat(r, z)

    def kappa(self, theta, z):
        return self.halo_norm*einasto.fast_Sigma(theta/self.Mpc2deg, self.r2, self.alpha) / self.S_cr(z)

    def m(self, theta, z):
        return (self.halo_norm*einasto.fast_M2d(theta, self.r2_ang, self.alpha) + self.mstar)/ (pi * self.S_cr(z) / self.Mpc2deg**2) 

    def alpha(self, theta, z):
        return (self.halo_norm/theta*einasto.fast_M2d(theta/self.r2_ang) + \
               self.mstar/pi*theta/(1./self.Mpc2deg)**2) / self.S_cr(z)

    def mu(self, theta, z):
        return ((1 - self.kappa(theta, z))**2 - self.gammat(theta, z)**2)**(-1)

    def rein(self, z, xtol=1e-6, xmin=1e-6, xmax=1.):
        bfunc = lambda theta: theta - self.alpha(theta, z)
        if bfunc(xmin)*bfunc(xmax) > 0:
            return 0.
        else:
            return brentq(bfunc, xmin, xmax, xtol=xtol)

    def gcompl(self, r, phi, z):
        return (1. - self.kappa(r, z))**(-1)*(self.gamma1(r, phi, z) + 1j*self.gamma2(r, phi, z))

    def gcomplstar(self, r, phi, z):
        return (1. - self.kappa(r, z))**(-1)*(self.gamma1(r, phi, z) - 1j*self.gamma2(r, phi, z))

    def source_gcompl(self):
        self.get_source_gammat()
        self.get_source_kappa()
        return (1. - self.sources['kappa'])**(-1) * self.sources['gammat'] * \
               (np.cos(2.*self.sources['phi']) + 1j*np.sin(2.*self.sources['phi']))

    def source_gamma(self):
        self.get_source_gammat()
        return self.sources['gammat'] * (np.cos(2.*self.sources['phi']) + 1j*np.sin(2.*self.sources['phi']))

    def get_image(self, y, z, rmin=None, rmax=10.):

        if rmin is None:
            rmin = self.rein(z)

        xfunc = lambda theta: theta - self.alpha(theta, z) - y

        return brentq(xfunc, rmin, rmax)

    def Sigma(self, theta): # surface mass density in M_Sun/pc^2
        return (self.halo_norm*einasto.fast_Sigma(theta/self.Mpc2deg, self.r2, self.alpha))/1e12

    def Sigmabar(self, theta):
        return (self.halo_norm*einasto.fast_M2d(theta, self.r2_ang, self.alpha)/(theta/self.Mpc2deg)**2/pi + \
                self.mstar/(theta/self.Mpc2deg)**2/pi) /1e12

    def DeltaSigma(self, theta):
        return (self.Sigmabar(theta) - self.Sigma(theta))

class AdcontrSersicCheat:
    # WARNING: CURRENT VERSION IS NOT SELF-CONSISTENT: ADIABATIC CONTRACTION IS CALCULATED ASSUMING A DEVAUCOULEURS PROFILE

    def __init__(self, z=0.3, m200=1e13, c200=5., mstar=1e11, reff=5., nser=4., nu=0., ra=0., dec=0., sources=None, \
                 cosmo=wl_cosmology.default_cosmo):

        import adcontr

        self.z = z
        self.cosmo = cosmo
        self.m200 = m200 # halo mass in M_Sun units
        self.mstar = mstar # stellar mass in M_Sun units
        self.fbar = self.mstar / (self.mstar + self.m200)
        self.rhocrit = wl_cosmology.rhoc(self.z, cosmo=self.cosmo)
        self.r200 = (self.m200*3./200./(4.*pi)/self.rhocrit)**(1/3.) #r200 in Mpc
        self.c200 = c200
        self.rs = self.r200/self.c200
        self.reff = reff # effective radius in kpc
        self.nser = nser # Sersic index
        self.nu = nu # adiabatic contraction efficiency parameter
        self.angD = wl_cosmology.Dang(self.z, cosmo=self.cosmo)
        self.Mpc2deg = np.rad2deg(1./self.angD)
        self.rs_ang = self.rs*self.Mpc2deg
        self.reff_ang = self.reff*self.Mpc2deg/1000.
        self.S_h = self.m200/self.rs**2
        self.S_bulge = self.mstar/self.reff**2*1e6
        self.sources = sources

        if ra is not None:
            self.ra = ra
        if dec is not None:
            self.dec = dec

    def update(self):
        self.r200 = (self.m200*3./200./(4.*pi)/self.rhocrit)**(1/3.)
        self.rs = self.r200/self.c200
        self.rs_ang = self.rs*self.Mpc2deg
        self.reff_ang = self.reff*self.Mpc2deg/1000.
        self.S_h = self.m200/self.rs**2
        self.S_bulge = self.mstar/self.reff**2*1e6
        self.fbar = self.mstar / (self.mstar + self.m200)

    def S_cr(self, z):
        Ds = wl_cosmology.Dang(z, cosmo=self.cosmo)
        Dds = wl_cosmology.Dang(z, self.z, cosmo=self.cosmo)
        return c**2/(4.*pi*G)*Ds/Dds/self.angD*Mpc/M_Sun

    def get_source_scr(self):
        nsource = len(self.sources['z'])
        s_crs = np.zeros(nsource)
        for i in range(nsource):
            if self.sources['z'][i] > self.z:
                s_crs[i] = self.S_cr(self.sources['z'][i])
        self.sources['s_cr'] = s_crs

    def get_source_polarcoords(self):

        self.sources['r'] = ((self.ra - self.sources['ra'])**2*np.cos(np.deg2rad(self.dec))**2 + (self.dec - self.sources['dec'])**2)**0.5

        xh =  - (self.sources['ra'] - self.ra)*np.cos(np.deg2rad(self.dec))
        yh = self.sources['dec'] - self.dec

        phih = np.arctan(yh/xh)

        phih[xh<0.] = phih[xh<0.] + np.pi
        phih[phih<0.] += 2.*np.pi

        self.sources['phi'] = phih

    def get_source_et(self):

        cosphi = np.cos(self.sources['phi'])
        sinphi = np.sin(self.sources['phi'])

        sin2phi = 2.*cosphi*sinphi
        cos2phi = cosphi**2 - sinphi**2

        self.sources['et'] = self.sources['e1']*cos2phi + self.sources['e2']*sin2phi

    def get_source_gammat(self):
        self.sources['gammat'] = (self.S_h/np.pi/(self.sources['r']/self.rs_ang)**2*adcontr.M2d(self.sources['r'], self.fbar, self.reff_ang, self.rs_ang, self.c200, self.nu) + \
        -self.S_h*adcontr.Sigma(self.sources['r'], self.fbar, self.reff_ang, self.rs_ang, self.c200, self.nu) + \
                self.S_bulge*(sersic.fast_M2d(self.sources['r']/self.reff_ang, self.nser)/(self.sources['r']/self.reff_ang)**2/pi - \
                              sersic.Sigma(self.sources['r']/self.reff_ang, self.nser, 1.))) / self.sources['s_cr']

    def get_source_kappa(self):
        self.sources['kappa'] = (self.S_h*adcontr.Sigma(sources['r'], self.fbar, self.reff_ang, self.rs_ang, self.c200, self.nu) + \
                self.S_bulge*sersic.Sigma(self.sources['r']/self.reff_ang, self.nser, 1.)) / self.sources['s_cr']

    def source_gcompl(self):
        self.get_source_gammat()
        self.get_source_kappa()
        return (1. - self.sources['kappa'])**(-1) * self.sources['gammat'] * \
               (np.cos(2.*self.sources['phi']) + 1j*np.sin(2.*self.sources['phi']))

    def source_gamma(self):
        self.get_source_gammat()
        return self.sources['gammat'] * (np.cos(2.*self.sources['phi']) + 1j*np.sin(2.*self.sources['phi']))


class GNFWdeV:

    def __init__(self, z=0.3, m200=1e13, c200=5., gammadm=1., mstar=1e11, reff=5., ra=0., dec=0., sources=None, \
                 cosmo=wl_cosmology.default_cosmo):

        self.z = z
        self.cosmo = cosmo
        self.m200 = m200 # halo mass in M_Sun units
        self.mstar = mstar # central point mass in M_Sun units
        self.rhocrit = wl_cosmology.rhoc(self.z, cosmo=self.cosmo)
        self.r200 = (self.m200*3./200./(4.*pi)/self.rhocrit)**(1/3.) #r200 in Mpc
        self.c200 = c200
        self.rs = self.r200/self.c200
        self.reff = reff # effective radius in kpc
        self.gammadm = gammadm
        self.angD = wl_cosmology.Dang(self.z, cosmo=self.cosmo)
        self.Mpc2deg = np.rad2deg(1./self.angD)
        self.rs_ang = self.rs*self.Mpc2deg
        self.reff_ang = self.reff*self.Mpc2deg/1000.
        self.halo_norm = self.m200/gnfw.fast_M3d(self.r200, self.rs, self.gammadm)
        self.S_bulge = self.mstar/self.reff**2*1e6
        self.sources = sources

        if ra is not None:
            self.ra = ra
        if dec is not None:
            self.dec = dec

    def update(self):
        self.r200 = (self.m200*3./200./(4.*pi)/self.rhocrit)**(1/3.)
        self.rs = self.r200/self.c200
        self.rs_ang = self.rs*self.Mpc2deg
        self.reff_ang = self.reff*self.Mpc2deg/1000.
        self.halo_norm = self.m200/gnfw.fast_M3d(self.r200, self.rs, self.gammadm)
        self.S_bulge = self.mstar/self.reff**2*1e6

    def S_cr(self, z): # critical surface mass density in M_Sun/Mpc^2
        Ds = wl_cosmology.Dang(z, cosmo=self.cosmo)
        Dds = wl_cosmology.Dang(z, self.z, cosmo=self.cosmo)
        return c**2/(4.*pi*G)*Ds/Dds/self.angD*Mpc/M_Sun

    def get_source_scr(self):
        nsource = len(self.sources['z'])
        s_crs = np.zeros(nsource)
        for i in range(nsource):
            if self.sources['z'][i] > self.z:
                s_crs[i] = self.S_cr(self.sources['z'][i])
        self.sources['s_cr'] = s_crs

    def get_source_polarcoords(self):

        self.sources['r'] = ((self.ra - self.sources['ra'])**2*np.cos(np.deg2rad(self.dec))**2 + (self.dec - self.sources['dec'])**2)**0.5

        xh =  - (self.sources['ra'] - self.ra)*np.cos(np.deg2rad(self.dec))
        yh = self.sources['dec'] - self.dec

        phih = np.arctan(yh/xh)

        phih[xh<0.] = phih[xh<0.] + np.pi
        phih[phih<0.] += 2.*np.pi

        self.sources['phi'] = phih

    def get_source_et(self):

        cosphi = np.cos(self.sources['phi'])
        sinphi = np.sin(self.sources['phi'])

        sin2phi = 2.*cosphi*sinphi
        cos2phi = cosphi**2 - sinphi**2

        self.sources['et'] = self.sources['e1']*cos2phi + self.sources['e2']*sin2phi

    def get_source_gammat(self):
        self.sources['gammat'] = (self.halo_norm*(gnfw.fast_M2d(self.sources['r'], self.rs_ang, self.gammadm)/pi/(self.sources['r']/self.Mpc2deg)**2 - gnfw.fast_Sigma(self.sources['r']/self.Mpc2deg, self.rs, self.gammadm)) + \
                self.S_bulge*(deVaucouleurs.fast_M2d(self.sources['r']/self.reff_ang)/(self.sources['r']/self.reff_ang)**2/pi - \
                deVaucouleurs.Sigma(self.sources['r']/self.reff_ang, 1.))) / self.sources['s_cr']

    def get_source_kappa(self):
        self.sources['kappa'] = (self.halo_norm*gnfw.fast_Sigma(self.sources['r']/self.Mpc2deg, self.rs, self.gammadm) + \
                self.S_bulge*deVaucouleurs.Sigma(self.sources['r']/self.reff_ang, 1.) )/ self.sources['s_cr']

    def gammat(self, theta, s_cr):
        return (self.halo_norm*(gnfw.fast_M2d(theta, self.rs_ang, self.gammadm)/pi/(theta/self.Mpc2deg)**2 - gnfw.fast_Sigma(theta/self.Mpc2deg, self.rs, self.gammadm)) + \
                self.S_bulge*(deVaucouleurs.fast_M2d(theta/self.reff_ang)/(theta/self.reff_ang)**2/pi - \
                              deVaucouleurs.Sigma(theta/self.reff_ang, 1.))) / s_cr

    def gamma1(self, r, phi, s_cr):
        return np.cos(2.*phi)*self.gammat(r, s_cr)

    def gamma2(self, r, phi, s_cr):
        return np.sin(2.*phi)*self.gammat(r, s_cr)

    def kappa(self, theta, s_cr):
        return (self.halo_norm*gnfw.fast_Sigma(theta/self.Mpc2deg, self.rs, self.gammadm) + \
                self.S_bulge*deVaucouleurs.Sigma(theta/self.reff_ang, 1.)) /s_cr

    def alpha(self, theta, s_cr):
        return (self.halo_norm/theta*gnfw.fast_M2d(theta/self.rs_ang) + \
               self.S_bulge/pi*theta/(theta/self.reff_ang)**2*deVaucouleurs.fast_M2d(theta/self.reff_ang)) \
               / s_cr

    def mu(self, theta, s_cr):
        return ((1 - self.kappa(theta, s_cr))**2 - self.gammat(theta, s_cr)**2)**(-1)

    def rein(self, s_cr, xtol=1e-6, xmin=1e-6, xmax=1.):
        bfunc = lambda theta: theta - self.alpha(theta, s_cr)
        if bfunc(xmin)*bfunc(xmax) > 0:
            return 0.
        else:
            return brentq(bfunc, xmin, xmax, xtol=xtol)

    def gcompl(self, r, phi, s_cr):
        return (1. - self.kappa(r, s_cr))**(-1)*(self.gamma1(r, phi, s_cr) + 1j*self.gamma2(r, phi, s_cr))

    def gcomplstar(self, r, phi, s_cr):
        return (1. - self.kappa(r, s_cr))**(-1)*(self.gamma1(r, phi, s_cr) - 1j*self.gamma2(r, phi, s_cr))

    def source_gcompl(self):
        self.get_source_gammat()
        self.get_source_kappa()
        return (1. - self.sources['kappa'])**(-1) * self.sources['gammat'] * \
               (np.cos(2.*self.sources['phi']) + 1j*np.sin(2.*self.sources['phi']))

    def source_gamma(self):
        self.get_source_gammat()
        return self.sources['gammat'] * (np.cos(2.*self.sources['phi']) + 1j*np.sin(2.*self.sources['phi']))

    def get_image(self, beta, s_cr, rmin=None, rmax=10.):

        if rmin is None:
            rmin = self.rein(z)

        xfunc = lambda theta: theta - self.alpha(theta, s_cr) - beta

        return brentq(xfunc, rmin, rmax)

    def Sigma(self, theta): # surface mass density in M_Sun/pc^2
        return (self.halo_norm*gnfw.fast_Sigma(theta/self.Mpc2deg, self.rs, self.gammadm) +\
                self.S_bulge*deVaucouleurs.Sigma(theta/self.reff_ang, 1.))/1e12

    def Sigmabar(self, theta):
        return (self.halo_norm*gnfw.fast_M2d(theta, self.rs_ang, self.gammadm)/(theta/self.Mpc2deg)**2/pi + \
                self.S_bulge*(deVaucouleurs.fast_M2d(theta/self.reff_ang)/(theta/self.reff_ang)**2/pi)) /1e12

    def DeltaSigma(self, theta):
        return (self.Sigmabar(theta) - self.Sigma(theta))


