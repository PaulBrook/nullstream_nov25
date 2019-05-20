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

#from nullstream_algebra import response_matrix
from nullstream_algebra import null_streams
import class_utils
# extra modules with functions for picking pulsars and picking sampling times
import _PTA_sim_pulsars, _PTA_sim_times, _PTA_sim_fourier, _PTA_sim_injections
from _PTA_sim_times import YEAR


@class_utils.add_functions_as_methods(_PTA_sim_pulsars.functions + 
                                      _PTA_sim_times.functions + 
                                      _PTA_sim_fourier.functions + 
                                      _PTA_sim_injections.functions)
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
