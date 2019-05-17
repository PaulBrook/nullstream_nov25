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
        self._freqs = 0
        self._signal = 0
        self._noise = 0
        self._signalFD = 0
        self._noiseFD = 0

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
        # loop through the pulsars
        nTOAs = self._times.shape[-1]
        noise = [rd.normal(scale=self._pulsars['rms'][i], size=nTOAs)
                 for i in range(self._n_pulsars)]

        self._noise = np.array(noise)
        assert(self._noise.shape == self._times.shape)
        # don't add to any previously existing noise
        self._noise = noise


    def plot_residuals(self):
        """
        Plot times vs residuals for all pulsars
        """
        fig, ax = plt.subplots(1)
        ax.plot(self._times.T, self.residuals.T, ls='none', marker='.')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('residuals (s)')

    def clear_residuals(self):
        """
        Delete any signal and noise from the residuals.
        """
        self._signal = np.zeros_like(self._times)
        self._noise = np.zeros_like(self._times)
        
    
    # we can make different functions like these to get different frequency spacings etc
    # FIXME maybe weights should also be computed here? (now commented out)
    def _linear_freqs(self, fmax=1e-7, alpha=1):
        """
        Choose linear grid of frequencies.
        """
        # set minimum frequency based on longest time span between any TOAs
        # with alpha some integer constant of order 1 or experiment with higher alpha
        Tmax = np.max(self._times) - np.min(self._times)
        fmin = 1 / (alpha * Tmax)
        # fmax chosen based on astrophysical prior, step same as fmin
        self._freqs = np.arange(fmin, fmax, step=fmin)
        
#        # compute weights per pulsar, then stack together to save in class
#        all_weights = []
#        for i in range(self._n_pulsars):
#
#            # get times that aren't nan
#            irregular_times = self._times[i][np.isfinite(self._times[i])]
#
#            weights = np.array([
#                            irregular_times[t_index+1]-irregular_times[t_index]
#                            if t_index<len(irregular_times)-1
#                            else 0
#                            for t_index in range(len(irregular_times)-1) ])
#            weights[-1] = irregular_times[-1] - np.sum(weights)
#
#            weights = np.ones(len(irregular_times))
#            
#            all_weights.append(weights)
#        
#        self._weights = np.vstack(all_weights)
            

    def fourier(self):
        """
        Compute the Fourier domain signal and noise.
        """
        # only implemented linear frequency spacing for now
        self._linear_freqs()
        
        self._signalFD = np.zeros((self._n_pulsars, len(self._freqs)), dtype=np.complex_)
        self._noiseFD = np.zeros((self._n_pulsars, len(self._freqs)), dtype=np.complex_)

        for i in range(self._n_pulsars):
            
            # get times that aren't nan
            irregular_times = self._times[i][np.isfinite(self._times[i])]

            weights = np.array([
                            irregular_times[t_index+1]-irregular_times[t_index]
                            if t_index<len(irregular_times)-1
                            else 0
                            for t_index in range(len(irregular_times)-1) ])
            weights[-1] = irregular_times[-1] - np.sum(weights)

            # FIXME lines above do nothing if we just set all weights to one
            weights = np.ones(len(irregular_times))
            
            #matrix m_ij of weights(t_i) * exp(-2*pi*i*f_j*t_i)
            M = np.array([[
                        weights[t_index] * np.exp(-2.*np.pi*(1j)*f*irregular_times[t_index])
                        for t_index in range(len(irregular_times))]
                        for f in self._freqs])

            # get signal without nans
            irregularly_sampled_signal = self._signal[i][np.isfinite(self._times[i])]
            self._signalFD[i] = np.dot(M, irregularly_sampled_signal)

            if np.shape(self._noise) is not ():
                irregularly_sampled_noise = self._noise[i][np.isfinite(self._times[i])]
                self._noiseFD[i] = np.dot(M, irregularly_sampled_noise)

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
    sim.evenly_sampled_times()

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

    # compute Fourier domain signal
    sim.fourier()

    # plot the Fourier domain residuals
    sim.plot_residuals_FD()
