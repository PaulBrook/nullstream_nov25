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
import os

import ptacake as cake
from ptacake.GW_models import sinusoid_TD

# all the potential numbers of pulsars
#Pvals = [3, 5, 8, 10, 15, 20, 30, 50, 100]
Pvals = [3, 5, 8 ,10, 20, 50, 100]
# maximum one
Pmax = 100

if __name__ == '__main__':
    
    runs_idx = 3
    
    directory = '/home/jgoldstein/Documents/projects/ptacake_runs/sky_runs{}'.format(runs_idx)
    pulsar_file = 'sky_runs{}_pulsars.csv'.format(runs_idx)
    radec_file = 'pulsars{}_radec_test.txt'.format(runs_idx)
    
    # pick a random seed
    # 1010 for sky_runs (1), 1011 for sky_runs2, 1012 for sky_runs3 etc
    seed = 1009 + runs_idx
    rd.seed(seed)
    
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
    # this will pick random start times for all the pulsars within 5 years
    max_t_start = 5 * cake.YEAR
    # all pulsars have the same end time
    t_end = 20 * cake.YEAR
    # set a gap on average every 5 years, lasting on average about 58 days
    exp_gap_spacing = 5 * cake.YEAR
    exp_gap_length = 5.0e6
    sim.randomized_times(mean_cadence=mean_dt, std_cadence=std_dt, min_cadence=min_dt,
                         max_t_start=max_t_start, t_end=t_end,
                         gaps=True, exp_gap_spacing=exp_gap_spacing, exp_gap_length=exp_gap_length,
                         seed=seed)
    
    
    ### choose injection parameters ###
    # args are: (times), phase, amplitude, polarization, cos(inclination), GW frequency
    true_args = [0.456, 1.0e-14, np.pi/7, 0.4, 2e-8]
    sim.inject_signal(sinusoid_TD, true_source, *true_args)
    
    
    sim.plot_residuals()
    
    ### compute S/N values ###
    # for each potential number of pulsars
    SNRs = []
    for P in Pvals:
        simtest = cake.PTA_sim()
        simtest.set_pulsars(sim._pulsars[['theta', 'phi']].values[:P], sim._pulsars['rms'].values[:P])
        simtest.randomized_times(mean_cadence=mean_dt, std_cadence=std_dt, min_cadence=min_dt,
                         max_t_start=max_t_start, t_end=t_end,
                         gaps=True, exp_gap_spacing=exp_gap_spacing, exp_gap_length=exp_gap_length,
                         seed=seed)
        simtest.inject_signal(sinusoid_TD, true_source, *true_args)
        
        snr = simtest.compute_snr()
        print('S/N at P={} {}'.format(P, snr))
        SNRs.append(snr)
        
        simtest.plot_pulsar_map(plot_point=true_source)
    
    ### save pulsars to file ###
    print('Saving pulsar file in {}'.format(directory))
    if not os.path.exists(directory):
        os.mkdir(directory)
    sim.pulsars_to_csv(os.path.join(directory, pulsar_file))
    
    ### save ra, dec and rms of pulsars for later plotting with ligo.skymap ###
    import pandas as pd
    pulsars = pd.read_csv(os.path.join(directory, pulsar_file))
    pulsars['Lon'] = (pulsars['phi'] + np.pi)%(2*np.pi)
    pulsars['Lat'] = np.pi/2 - pulsars['theta']
    pulsars['Lon_deg'] = (180/np.pi) * pulsars['Lon']
    pulsars['Lat_deg'] = (180/np.pi) * pulsars['Lat']
    np.savetxt(os.path.join(directory, radec_file), pulsars[['Lon_deg', 'Lat_deg', 'rms']].values)    
    
#    sim.fourier_residuals()
#    sim.plot_residuals_FD()
#    
#    sim.white_noise()
#    sim.plot_residuals()
#    sim.fourier_residuals()
#    sim.plot_residuals_FD()
#    