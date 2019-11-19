#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:11:27 2019

@author: jgoldstein

want to run: bunch of cpnest runs using FD and FDnull_only (maybe also TD?) likelihoods
on sims with increasing numbers of pulsars, with a fixed source. Obtain posteriors
on the sky location (theta, phi) parameters.
This script is to choose some set of pulsars for these runs and to investigate
what a good example source would be (perhaps S/N ~10 for P=5?)
"""

import numpy as np
import numpy.random as rd

import ptacake as cake
from ptacake.GW_models import sinusoid_TD

# all the potential numbers of pulsars
Pvals = [3, 5, 8, 10, 15, 20, 30, 50, 100]
# maximum one
Pmax = 100

if __name__ == '__main__':
    
    # pick a random seed
    seed = 1010
    
    ### create a sim                                     ###
    ### and pick 100 random pulsars (with IPTA sky bias) ###
    
    sim = cake.PTA_sim()
    sim.random_pulsars(Pmax, mean_rms=1e-7, sig_rms=4e-8, min_rms=1e-8, 
                       uniform=False, seed=seed)

    # pick a source location for the injection
    # not random because it makes nicer plots if the source is in the middle of the sky map
    true_source = [0.456 * np.pi, 0.321]
    
    sim.plot_pulsar_map(plot_point=true_source)
    
    
    ### set unevenly sampled, gappy times ###
    
    mean_dt = 2.0e6 # about 23 days
    std_dt = 1.0e5
    min_dt = 1.0e5 # about 1.2 days
    # pick random start times for all the pulsars within 5 years
    t_start = rd.random(size=Pmax) * 5 * cake.YEAR
    # all pulsars have the same end time
    t_end = 20 * cake.YEAR
    # set a gap on average every 5 years, lasting on average about 58 days
    exp_gap_spacing = 5 * cake.YEAR
    exp_gap_length = 5.0e6
    sim.randomized_times(mean_cadence=mean_dt, std_cadence=std_dt, min_cadence=min_dt,
                         t_start=t_start, t_end=t_end,
                         gaps=True, exp_gap_spacing=exp_gap_spacing, exp_gap_length=exp_gap_length,
                         seed=seed)
    
    
    ### choose injection parameters ###
    # args are: (times), phase, amplitude, polarization, cos(inclination), GW frequency
    true_args = [0.123, 1.0e-14, np.pi/7, 0.4, 2e-8]
    sim.inject_signal(sinusoid_TD, true_source, *true_args)
    
    
    sim.plot_residuals()
    
    ### compute S/N values ###
    # for each potential number of pulsars
    SNRs = []
    for P in Pvals:
        simtest = cake.PTA_sim()
        simtest.set_pulsars(sim._pulsars[['theta', 'phi']].values[:P], sim._pulsars['rms'].values[:P])
        simtest.randomized_times(mean_cadence=mean_dt, std_cadence=std_dt, min_cadence=min_dt,
                         t_start=t_start[:P], t_end=t_end,
                         gaps=True, exp_gap_spacing=exp_gap_spacing, exp_gap_length=exp_gap_length,
                         seed=seed)
        simtest.inject_signal(sinusoid_TD, true_source, *true_args)
        
        snr = simtest.compute_snr()
        print('S/N at P={} {}'.format(P, snr))
        SNRs.append(snr)
    
#    sim.fourier_residuals()
#    sim.plot_residuals_FD()
#    
#    sim.white_noise()
#    sim.plot_residuals()
#    sim.fourier_residuals()
#    sim.plot_residuals_FD()
#    