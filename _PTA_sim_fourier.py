#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:22:15 2019

@author: jgoldstein

functions to go in the main PTA_sim class that have to do with funky fouriers
"""

import numpy as np
import matplotlib.pyplot as plt


# we can make different functions like these to get different frequency spacings etc

def _weights_matrix(self, times, freqs):
    # append ensures we get n diff values if there are n times (instead of n-1)
    weights = np.diff(times, append=times[-1])
    # FIXME what should the last weight be??? 
    weights[-1] = times[-1] - np.sum(weights)
        
    #matrix m_ij of weights(t_i) * exp(-2*pi*i*f_j*t_i)
    expanded_freqs = np.expand_dims(self._freqs, axis=-1)
    mat = weights * np.exp(-2*np.pi*(1j)*expanded_freqs*times)
    return weights, mat

def _setup_TOAs_fourier(self, fmax=1e-7, alpha=1):
    """
    Set up for the funky fourier on the TOA residuals with unevenly sampled times.
    
    Chooses a regular grid of frequencies with fmin = 1/(alpha * Tmax), where
    Tmax is the maximum time between any two TOAs in the data set. The frequency
    spacing is also fmin. Compute the weights as w(t_j)  = t_j+1 - t_j, with
    w_n = 0??? the last weight, for a pulsar with n TOAs. Then the fourier
    matrix is m_jk = w(t_j) * exp(-2*pi*i*f_k*t_j), with j TOAs and k frequencies.
    """
    # set minimum frequency based on longest time span between any TOAs
    # with alpha some integer constant of order 1 or experiment with higher alpha
    all_times_not_nan = self._times[np.isfinite(self._times)]
    Tmax = np.max(all_times_not_nan) - np.min(all_times_not_nan)
    fmin = 1 / (alpha * Tmax)
    # fmax chosen based on astrophysical prior, step same as fmin
    self._freqs = np.arange(fmin, fmax+fmin, step=fmin)
    
    # create empty fourier domain signal and noise of the right shape
    self._signalFD = np.zeros((self._n_pulsars, len(self._freqs)), dtype=np.complex_)
    self._noiseFD = np.zeros((self._n_pulsars, len(self._freqs)), dtype=np.complex_)
    
    # compute weights and fourier matrix per pulsar
    self._TOA_weights = []
    self._TOA_fourier_mats = []
    for p in range(self._n_pulsars):
        
        # get times that aren't nan
        irregular_times = self._times[p][np.isfinite(self._times[p])]
        weights, mat = self._weights_matrix(irregular_times, self._freqs)
        self._TOA_weights.append(weights)
        self._TOA_fourier_mats.append(mat)
    
    # we have everything set up to fourier the TOA residuals now
    self._TOA_fourier_ready = True
    
def _setup_model_fourier(self):
    """
    Set up for the funky fourier on the model, with same frequencies as the TOA residuals.
    """
    # we need the frequencies from the TOA residuals fourier setup
    if not self._TOA_fourier_ready:
        self._setup_TOAs_fourier()
        
    # pick a set of times to sample the model at. 
    # The maximum frequency gives us a cadence
    step = 1 / (2 * np.max(self._freqs))
    # start before the first TOA and end after the last
    all_times_not_nan = self._times[np.isfinite(self._times)]
    t_min = np.min(all_times_not_nan) - step
    t_max = np.max(all_times_not_nan) + step
    self._model_times = np.arange(t_min, t_max, step=step)
    
#    ## TEST same model times as evenly sampled TOA times
#    self._model_times = self._times[0]
    
    weights, mat = self._weights_matrix(self._model_times, self._freqs)
    self._model_weights = weights
    self._model_fourier_mat = mat
    
    self._model_fourier_ready = True

def fourier_residuals(self):
    """
    Compute the Fourier domain signal and noise.
    """
    if not self._TOA_fourier_ready:
        print("First time fourier of TOAs, setting up...")
        self._setup_TOAs_fourier()

    for i in range(self._n_pulsars):
        
        fourier_M = self._TOA_fourier_mats[i]

        # get signal without nans
        irregularly_sampled_signal = self._signal[i][np.isfinite(self._times[i])]
        self._signalFD[i] = np.dot(fourier_M, irregularly_sampled_signal)

        if np.shape(self._noise) is not ():
            irregularly_sampled_noise = self._noise[i][np.isfinite(self._times[i])]
            self._noiseFD[i] = np.dot(fourier_M, irregularly_sampled_noise)
    
            
def fourier_model(self, model, *args, **kwargs):
    """
    Get funky fourier of the model (at previously set times).
    
    Parameters
    ----------
    model: tuple of arrays (hplus, hcross) or callable
        hplus, hcross must be sampled at previously set times (PTA_sim._model_times)
        if callable, model(PTA_sim._model_times, *args, **kwargs) must return
        a tuble of (hplus, hcross) like above
    *args, **kwargs
        passed on to model if callable
    
    Returns
    -------
    tuple of numpy arrays
        fourier domain (hplus, hcross), at frequencies in PTA_sim._freqs
    """
    if not self._model_fourier_ready:
        print("First time fourier of model, setting up...")
        self._setup_model_fourier()

    if hasattr(model, '__call__'):
        hplus, hcross = model(self._model_times, *args, **kwargs)
    else:
        hplus, hcross = model
    assert hplus.shape == self._model_times.shape
    assert hcross.shape == self._model_times.shape
    
    fourier_hplus = np.dot(self._model_fourier_mat, hplus)
    fourier_hcross = np.dot(self._model_fourier_mat, hcross)
    return fourier_hplus, fourier_hcross
    

def plot_residuals_FD(self):
    """
    Plot times vs residuals for all pulsars
    """
    res = self.residualsFD
    fig, ax = plt.subplots(1)
    for i in range(self._n_pulsars):
        ax.plot(np.log10(self._freqs), np.log10(abs(res[i])), ls='-', marker='.')
    ax.set_xlabel('log_10( frequency (Hz) )')
    ax.set_ylabel('log_10( | residualsFD (s^2) | )')
    return fig


functions = [_weights_matrix, _setup_TOAs_fourier, _setup_model_fourier, 
             fourier_residuals, fourier_model, plot_residuals_FD]
