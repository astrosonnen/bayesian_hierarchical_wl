# Bayesian hierarchical galaxy-galaxy lensing

This Python code allows to carry out galaxy-galaxy lensing measurements with the Bayesian hierarchical inference formalism of Sonnenfeld & Leauthaud [(2018)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.477.5460S/abstract).

### Requirements
- emcee
- h5py
- A shape catalog

To use, add this directory to your `PYTHONPATH` environment variable.

## The method in a nutshell

Please refer to [this](method_description.ipynb) notebook.

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

### Step 2: evaluate the weak lensing likelihood on a grid

The script `examples/get_wl_likelihood_grids.py` loops over the lenses and calculates the likelihood of the observed shape measurements given the model, on a grid of values of halo mass, concentration and stellar mass. This script takes a few hours to run as it is. You might want to split the sample of lenses and run it in parallel.

### Step 3: infer the hyper-parameters

The script `examples/infer_nfw_shmr.py` fits for the hyper-parameters describing the SHMR and the concentration distribution of the lens sample. 
The model assumes a Gaussian distribution in log stellar mass, a Gaussian distribution in log halo mass with mean that scales with stellar mass, a Gaussian distribution in log concentration.
The marginalization over individual lens parameters is carried out by Monte Carlo integration, using the grids of the weak lensing likelihood obtained previously.
This script should take a few hours to run. It can be made faster by running emcee in multi-thread mode: you'll need to add `threads=Nthread` while calling `emcee.EnsampleSampler` in line 155. 
The end product is an .hdf5 file containing the MCMC chain, as returned by emcee.


