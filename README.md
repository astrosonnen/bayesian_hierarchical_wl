# Bayesian hierarchical galaxy-galaxy lensing

This Python code allows to carry out galaxy-galaxy lensing measurements with the Bayesian hierarchical inference formalism of Sonnenfeld & Leauthaud (2018).

### Requirements
- pymc
- emcee
- h5py
- A shape catalog

To use, add this directory to your `PYTHONPATH` environment variable.

## Example 1: SHMR and concentration of massive quiescent galaxies from the SDSS legacy sample

Let us fit for the stellar-to-halo mass relation and halo concentration distribution (mean and scatter of $c_{200}$) for a stellar mass-limited sample of galaxies from the SDSS spectroscopic sample.
This is done in three steps: 
1. Make a catalog of background sources around the lens galaxies in the sample.
2. For each lens, fit a composite stellar + halo model to the shape measurements, assuming an NFW profile for the halo and an interim prior on the model parameters.
3. Run an MCMC chain to sample the posterior probability distribution of the model hyper-parameters (average halo mass, stellar-halo mass correlation, scatter, etc.), marginalizing over the parameters of individual lenses.

### Step 1: assign background sources to the lenses

The file `examples/sdss_legacy_hscoverlap_mcut11.0.cat` is a catalog of massive quiescent galaxies from the SDSS legacy spectroscopic sample. It includes stellar masses, obtained from the MPA-JHU catalog, corrected to a Chabrier IMF. The sample is the same used in [Sonnenfeld et al. (2018)](https://arxiv.org/abs/1801.01883).

The script `examples/assign_sources.py` reads the shape measurement catalog and matches background sources to the galaxies in the lens sample. (This repository does NOT provide the shape catalog). 

This script is meant to be run on shape measurements from the internal data release of the HSC 16a shape catalog. If you don't have access to the HSC collaboration data products, you will need to modify substantially the code in order for it to work with your shape catalog.

### Step 2: sample individual lenses assuming an interim prior

The script `examples/sample_individ_nfw.py` loops over the lenses and fits a NFW + point source model to the stellar mass and shape measurements. The output is a series of .hdf5 files, one for each lens, containing the MCMC chain of the posterior probability distribution of the model parameters ($M_{200}$, $c_{200}$, $M_*$) given the data and the interim prior.
In this case, the interim prior is set to a Gaussian with a relatively large dispersion. When looked at individually, these chains are not very meaninfgul, as they are mostly determined by the interim prior.
This script takes a few hours to run as it is. You might want to split the sample of lenses and run it in parallel.

### Step 3: infer the hyper-parameters

The script `examples/infer_nfw_shmr.py` selects a subsample of lenses within a redshift interval, then fits for the hyper-parameters describing the SHMR and the concentration distribution of the sample. This is done while marginalizing over the values of stellar mass, halo mass and concentration of individual lenses. This marginalization is carried out by Monte Carlo integration and importance sampling, using the MCMC chains of individual lenses obtained previously.
This script should take around one hour to run. The end product is a pickled object containing the MCMC chain, as returned by emcee.

