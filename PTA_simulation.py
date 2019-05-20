#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:24:19 2019

@author: jgoldstein
"""

import numpy as np
import numpy.random as rd
import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt

try:
    from jannasutils import radec_location_to_ang, isIterable
except:
    # use hacked excerpt from jannasutils
    from from_jannasutils import radec_location_to_ang, isIterable

from nullstream_algebra import response_matrix
import class_utils
# extra modules with functions for picking pulsars and picking sampling times
import _PTA_sim_pulsars, _PTA_sim_times
from _PTA_sim_times import YEAR


@class_utils.add_functions_as_methods(_PTA_sim_pulsars.functions + _PTA_sim_times.functions)
class PTA_sim:
    def __init__(self):
        self._pulsars = pd.DataFrame(columns=['theta', 'phi', 'rms', 'nTOA'])
        self._n_pulsars = 0
        self._times = 0
        self._signal = 0
        self._noise = 0
        
        # fourier stuff
        self._TOA_fourier_ready = False
        self._model_fourier_ready = False
        self._signalFD = 0
        self._noiseFD = 0
        self._freqs = 0
        self._TOA_weights = []
        self._TOA_fourier_mats = []
        self._model_weights = []
        self._model_fourier_mat = []

    @property
    def residuals(self):
        return self._signal + self._noise

    @property
    def residualsFD(self):
        return self._signalFD + self._noiseFD

    def inject_signal(self, signal_func, source, *signal_args, **signal_kwargs):
        """
        Inject signal into the residuals.
        """
        hplus, hcross = signal_func(self._times, *signal_args, **signal_kwargs)
        responses = response_matrix(*source, self._pulsars[['theta', 'phi']].values)
        Fplus = np.expand_dims(responses[:, 0], -1)
        Fcross = np.expand_dims(responses[:, 1], -1)
        
        # add to current signal in case we want multiple injections done
        self._signal += Fplus * hplus + Fcross * hcross

    def white_noise(self):
        """
        Inject gaussian noise according to each pulsar's rms level.
        This deletes any previously injected noise (but keeps signal the same).
        """
        # annoyingly, rd.normal cannot handle both an array for the scale values
        # and more than a scalar output for each of those values, so we have to
        # loop through the 
        # ---> it can, but only in the reverse order (so the scale shape is the
        # same as the last, not the first axis). So we transpose at the end.
        reverse_shape = self._times.T.shape
        noise = rd.normal(scale=self._pulsars['rms'].values, size=reverse_shape)
        # don't add to any previously existing noise
        self._noise = noise.T

        assert(self._noise.shape == self._times.shape)


    def plot_residuals(self):
        """
        Plot times vs residuals for all pulsars
        """
        fig, ax = plt.subplots(1)
        ax.plot(self._times.T, self.residuals.T, ls='none', marker='.')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('residuals (s)')

#    def clear_residuals(self):
#        """
#        Delete any signal and noise from the residuals.
#        """
#        self._signal = np.zeros_like(self._times)
#        self._noise = np.zeros_like(self._times)
        
        
    # TODO
    # DONE simplify the bit of code that computes the weights (should be doable with diff)
    #
    # DONE move weight and fourier matrix computation to _linear_freqs or similar function
    # because these things can be precomputed (then fourier applies it to whatever quantity)
    # 
    # DONE setup choosing frequencies and precomputing stuff for the model (somewhat more densely
    # sampled than the data, can be evenly sampled)
    #
    # ... likelihood, cpnest etc etc
    
    # we can make different functions like these to get different frequency spacings etc
    # FIXME maybe weights should also be computed here? (now commented out)
        
    def _weights_matrix(self, times, freqs):
        # append ensures we get n diff values if there are n times (instead of n-1)
        weights = np.diff(times, append=times[-1])
        # FIXME what should the last weight be??? 
        weights[-1] = times[-1] - np.sum(weights)
            
        #matrix m_ij of weights(t_i) * exp(-2*pi*i*f_j*t_i)
        expanded_freqs = np.expand_dims(sim._freqs, axis=-1)
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


if __name__ == '__main__':
    rd.seed(1234)
    
    print('An example of PTA sim')
    # make a simulation object (we may want to have initialisation options that
    # automatically do the next few steps, but for now we do them by hand)
    Npsr = 5
    sim = PTA_sim()

    # make some pulsars, in this case 5 random ones with some variation in rms
    # and plot a skymap (bigger markers are better pulsars)
    sim.random_pulsars(Npsr, sig_rms=5e-8)
    sim.plot_pulsar_map()

    # set some evenly sampled times (default options)
    #sim.evenly_sampled_times()
     
    # randomized times
    sim.randomized_times()

    # generate some (very) unevenly sampled times
#    mean_cadences = 10**np.random.normal(6, 0.5, Npsr) # lognormal
#    t_starts = np.random.rand(Npsr) * 10 * YEAR
#    exp_gap_spacings = 10**np.random.normal(0.5, 0.5, Npsr) * YEAR
#
#    sim.gappy_times(mean_cadences=mean_cadences, t_starts=t_starts,
#                    exp_gap_spacings = exp_gap_spacings)

    # inject a sinusoid signal
    # arguments are: phase, amplitude, polarization, cos(i), GW frequency (rd/s)
    from GW_models import sinusoid_TD
    GW_args = [0.1, 1e-12, np.pi/7, 0.3, 4e-8]
    source = (0.8 * np.pi, 1.3 * np.pi)
    sim.inject_signal(sinusoid_TD, source, *GW_args)

    # inject white noise
    sim.white_noise()

    # plot the residuals
    sim.plot_residuals()

    # compute Fourier domain residuals
    sim.fourier_residuals()

    # plot the Fourier domain residuals
    sim.plot_residuals_FD()
