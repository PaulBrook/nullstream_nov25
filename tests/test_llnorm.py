#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 12:23:03 2019

@author: jgoldstein

test/make plots of normalizations of the log likelihoods
"""

import unittest
import numpy as np
import numpy.random as rd
import numpy.testing as npt
import matplotlib.pyplot as plt

from ptacake import PTA_sim, YEAR
from ptacake.GW_models import sinusoid_TD

class Test_likelihood(unittest.TestCase):
    
    ### Set plotting to True here if you want this unit test to make plots
    def __init__(self, *args, **kwargs):
        super(Test_likelihood, self).__init__(*args, **kwargs)
        self.plotting = False
        
    @classmethod
    def setUpClass(cls):
        
        #cls.sim = setup_evenly_sampled(n_pulsars=5, seed=1234567)
        rd.seed(1234567)
        cls.sim = PTA_sim()
        
        cls.sim.random_pulsars(5, sig_rms=5e-8)
        # bit of a long Dt and short time, so that we don't have too many points
        # and tests run relatively quickly
        Dt = 1.2e6
        T = 7*YEAR
        t_start = 0
        cls.sim.evenly_sampled_times(cadence=Dt, T=T, t_start=t_start)
        
        # make a sinusoidal signal
        # parameters past times are: phase, amplitude, polarization, cos(i), GW angular freq
        sinusoid_args = [0.123, 1e-14, np.pi/7, 0.5, 2e-8]
        # choose source (theta, phi) coordinates
        source = (0.8*np.pi, 1.3*np.pi)
        cls.sim.inject_signal(sinusoid_TD, source, *sinusoid_args)
        
#        # for test purposes, use np fft frequencies
        freqs = np.fft.rfftfreq(len(cls.sim._times[0]), d=(cls.sim._times[0][1] - cls.sim._times[0][0]))
#        cls.sim.fourier_residuals(overwrite_freqs = freqs)
#       
        # test with fft instead of matrix ft
        cls.sim.fft_residuals()
        
        # for test purposes, use the same times for residuals and model
        # (so no oversampling for the model times)
        cls.sim._setup_model_fourier(overwrite_times=cls.sim._times[0])
        
        npt.assert_equal(cls.sim._freqs, freqs)
        
        # make concatenated residuals for null-stream likelihood
        cls.sim.concatenate_residuals()
        
        
        cls.ll_funcs = {'TD':cls.sim.log_likelihood_TD, 
                         'FD':cls.sim.log_likelihood_FD,
                         'TD_ns':cls.sim.log_likelihood_TD_ns,
                         'FD_ns':cls.sim.log_likelihood_FD_ns}
        
        cls.num_points = 50
        cls.param_idxs_ranges = {'theta': (0, np.linspace(0, np.pi, num=cls.num_points)),
                                  'phi': (1, np.linspace(0, 2*np.pi, num=cls.num_points)),
                                  'phase': (2, np.linspace(0, 2*np.pi, num=cls.num_points)),
                                  'amp': (3, 10**np.linspace(-16, -13.5, num=cls.num_points)),
                                  'amplitude': (3, 10**np.linspace(-16, -13.5, num=cls.num_points)),
                                  'pol': (4, np.linspace(0, np.pi, num=cls.num_points)),
                                  'polarization': (4, np.linspace(0, np.pi, num=cls.num_points)),
                                  'cosi': (5, np.linspace(-1, 1, num=cls.num_points)),
                                  'cosinclination': (5, np.linspace(-1, 1, num=cls.num_points)),
                                  'freq': (6, np.linspace(5e-9, 1e-7, num=cls.num_points)),
                                  'GWfreq': (6, np.linspace(5e-9, 1e-7, num=cls.num_points)),
                                  'frequency': (6, np.linspace(5e-9, 1e-7, num=cls.num_points)),
                                  'GWfrequency': (6, np.linspace(5e-9, 1e-7, num=cls.num_points))}
        
        cls.test_params=['theta', 'phi', 'phase', 'amp', 'pol', 'cosi', 'GWfreq']
        cls.num_params = len(cls.test_params)

    def ll_vs_param(self, ll_name, param_name):

        ll_func = self.ll_funcs[ll_name]
        
        # including theta, phi at the start
        standard_args = [0.8*np.pi, 1.3*np.pi, 0.123, 1e-14, np.pi/7, 0.5, 2e-8]
        
        param_idx, param_range = self.param_idxs_ranges[param_name]
        
        norms = np.zeros(len(param_range))
        for i, param in enumerate(param_range):
            test_args = standard_args.copy()
            test_args[param_idx] = param
            test_source = test_args[:2]
            test_sinusoid_args = test_args[2:]
            norms[i] = ll_func(test_source, sinusoid_TD, test_sinusoid_args, return_only_norm=True)
            
        return param_range, norms
    
    def ll_all_params(self, ll_name):
        
        if self.plotting:
            fig = plt.figure(figsize=(5, 11))
            
        for i, param_name in enumerate(self.test_params):
            param_range, norms = self.ll_vs_param(ll_name, param_name)
            
            if self.plotting:
                # plot param vs norm
                ax = fig.add_subplot(self.num_params, 1, i+1)
                ax.plot(param_range, norms)
                ax.set_xlabel(param_name)
                ax.set_ylabel('{} norm'.format(ll_name))
                
            #test all norms are equal
            # except theta, phi in ns case
            if 'ns' in ll_name and param_name in ['theta', 'phi']:
                pass
            else:
                np.all(norms == norms[0])
            
        if self.plotting:
            fig.tight_layout()
            figname = '{}_norm.pdf'.format(ll_name)
            fig.savefig(figname.replace(' ', '_'))
        
    
    def test_TD_norms(self):
        self.ll_all_params('TD')
    
    def test_TD_ns_norm(self):
        self.ll_all_params('TD_ns')
        
    def test_FD_norm(self):
        self.ll_all_params('FD')
        
    def test_FD_ns_norm(self):
        self.ll_all_params('FD_ns')
 
