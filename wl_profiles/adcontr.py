import numpy as np
import pickle


grid_dir = os.environ.get('BHWLDIR') + '/wl_profiles/'

f = open(gdir_dir+'/adcontr_m2d_coarsendinterp.dat', 'r')
adcontr_m2d_interp = pickle.load(f)
f.close()

f = open(gdir_dir+'/adcontr_sigma_coarsendinterp.dat', 'r')
adcontr_sigma_interp = pickle.load(f)
f.close()

def M2d(r, fbar, reff, rs, c200, nu):

    r = np.atleast_1d(r)
    l = len(r)

    point = np.zeros((l, 5))
    point[:, 0] = np.log10(fbar) * np.ones(l)
    point[:, 1] = np.log10(reff/rs) * np.ones(l)
    point[:, 2] = np.log10(c200) * np.ones(l)
    point[:, 3] = nu * np.ones(l)

    point[:, 4] = r/rs

    return adcontr_m2d_interp.eval(point)

def Sigma(r, fbar, reff, rs, c200, nu):

    r = np.atleast_1d(r)
    l = len(r)

    point = np.zeros((l, 5))
    point[:, 0] = np.log10(fbar) * np.ones(l)
    point[:, 1] = np.log10(reff/rs) * np.ones(l)
    point[:, 2] = np.log10(c200) * np.ones(l)
    point[:, 3] = nu * np.ones(l)

    point[:, 4] = r/rs

    return adcontr_sigma_interp.eval(point)

