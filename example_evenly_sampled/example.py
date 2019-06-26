"""
Created on Wed Jun 26 11:11:23 2019

@author: jgoldstein

example of ptacake, also making some plots
"""
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

import ptacake as cake
from ptacake.GW_models import sinusoid_TD

# change to the directory you want to store plots
DIR = '/home/jgoldstein/Documents/projects/Nullstream/example_evenly_sampled/'

if __name__ == "__main__":
    np.random.seed(2345)
    
    # make a PTA_sim with some pulsars
    sim = cake.PTA_sim()
    sim.random_pulsars(5)
    pulsar_map = sim.plot_pulsar_map()
    pulsar_map.savefig(join(DIR, 'pulsar_map.pdf'))
    
    # get evenly sampled times and inject a sinusoid signal into the residuals
    sim.evenly_sampled_times()
    source = (0.8*np.pi, 1.3*np.pi)
    sinusoid_args = [0.123, 1e-16, np.pi/7, 0.5, 2e-8]
    sim.inject_signal(sinusoid_TD, source, *sinusoid_args)
    TD_residuals_fig = sim.plot_residuals()
    TD_residuals_fig.savefig(join(DIR, 'TD_residuals.pdf'))
    
    # use funky fourier to get the FD residuals
    sim.fourier_residuals()
    FD_residuals_fig = sim.plot_residuals_FD()
    FD_residuals_fig.savefig(join(DIR, 'FD_residuals.pdf'))

    # make plots of variable vs likelihood
    var_indices = {'phase':0,
                   'amplitude':1,
                   'polarization':2,
                   'cosi':3,
                   'frequency':4}
    var_ranges = {'phase':[0, 2*np.pi],
                  'amplitude':[1e-20, 10**(-15.5)],
                  'polarization':[0, np.pi],
                  'cosi':[-1, 1],
                  'frequency':[1e-9, 1e-7]}
    
    for var in var_indices.keys():
        print('making plot for {} vs log like'.format(var))
        r = var_ranges[var]
        
        if var in ['amplitude', 'frequency']:
            var_values = np.logspace(np.log10(r[0]), np.log10(r[1]), num=1000)
        else:
            var_values = np.linspace(r[0], r[1], num=1000)
    
        log_likes = []
        for x in var_values:
            test_args = sinusoid_args.copy()
            test_args[var_indices[var]] = x
            ll = sim.log_likelihood_FD_ns(source, sinusoid_TD, test_args)
            log_likes.append(ll)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if var in ['amplitude', 'frequency']:
            ax.set_xscale('log')
        ax.plot(var_values, log_likes)
        true_x = sinusoid_args[var_indices[var]]
        miny, maxy = ax.get_ylim()
        ax.vlines(true_x, miny, maxy, color='k', linestyle='--')
        
        ax.set_xlabel(var)
        ax.set_ylabel('log likelihood')
        fig.savefig(join(DIR, '{}_vs_log_like.pdf'.format(var)))
    
    
    

    

