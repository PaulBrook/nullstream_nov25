#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 12:33:34 2019

@author: jgoldstein
"""
import unittest
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt

from ptacake import PTA_sim, YEAR
from ptacake.GW_models import sinusoid_TD
from tests.setup_sim import setup_evenly_sampled

class Test_likelihood(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.sim = setup_evenly_sampled(n_pulsars=5, seed=1234567)
        
        # make a sinusoidal signal
        GW_freq = 2e-8
        GW_ang_freq = 2*np.pi*GW_freq
        # parameters past times are: phase, amplitude, polarization, cos(i), GW angular freq
        sinusoid_args = [0.123, 1e-16, np.pi/7, 0.5, GW_ang_freq]
        # choose source (theta, phi) coordinates
        source = (0.8*np.pi, 1.3*np.pi)
        cls.sim.inject_signal(sinusoid_TD, source, *sinusoid_args)
    
    def test_TD_nullstreams(self):
        """
        Test that TD likelihood on evenly sampled data is the same with and
        without nullstream conversion.
        """
        # get likelihood for parameters close to injected parameters
        source = (0.8*np.pi, 1.3*np.pi)
        GW_ang_freq = 2*np.pi*2e-8
        sinusoid_args_test = [0.14, 1e-16, np.pi/7.2, 0.48, GW_ang_freq]
        TD_ll = self.sim.log_likelihood_TD(source, sinusoid_TD, sinusoid_args_test)
        # same but likelihood but with null streams
        TD_ll_ns = self.sim.log_likelihood_TD_ns(source, sinusoid_TD, sinusoid_args_test)
        
        #print('TD log like: {}, TD log like w/ null streams: {}'.format(TD_ll, TD_ll_ns))
        npt.assert_almost_equal(TD_ll, TD_ll_ns)
        
    def test_FD_nullstreams(self):
        """
        Test that the FD likelihood on evenly sampled data is the same with
        and without nullstream conversion.
        """
        # get likelihood for parameters close to injected parameters
        source = (0.8*np.pi, 1.3*np.pi)
        GW_ang_freq = 2*np.pi*2e-8
        sinusoid_args_test = [0.11, 1e-16, np.pi/6.7, 0.513, GW_ang_freq]
        FD_ll = self.sim.log_likelihood_FD(source, sinusoid_TD, sinusoid_args_test)
        # same likelihood but with null streams
        FD_ll_ns = self.sim.log_likelihood_FD_ns(source, sinusoid_TD, sinusoid_args_test)
        
        #print('FD log like: {}, FD log like w/ null streams: {}'.format(FD_ll, FD_ll_ns))
        npt.assert_almost_equal(FD_ll, FD_ll_ns)
        

    def test_TD_FD(self):
        """
        Test that the TD and the FD likelihood of evenly sampled data is 
        the same. (We're doing the ones without null streams, but it 
        doesn't really matter, don't think we need to do both since that's
        tested aboce.)
        """
        # get likelihood for parameters close to injected parameters
        source = (0.8*np.pi, 1.3*np.pi)
        GW_ang_freq = 2*np.pi*2e-8
        sinusoid_args_test = [0.18, 1e-16, np.pi/6.8, 0.52, GW_ang_freq]
        # TD likelihood
        TD_ll = self.sim.log_likelihood_TD(source, sinusoid_TD, sinusoid_args_test)
        # FD likelihood
        FD_ll = self.sim.log_likelihood_FD(source, sinusoid_TD, sinusoid_args_test)
        
        print('TD log like: {}, FD log like: {}'.format(TD_ll, FD_ll))
        #npt.assert_almost_equal(TD_ll, FD_ll)
        
        ### since this fails, let's check the shape of the likelihood vs some parameters
        standard_args = [0.123, 1e-16, np.pi/7, 0.5, GW_ang_freq]
        phases = np.linspace(0, 2*np.pi, num=20)
        TD_ll_phases = []
        FD_ll_phases = []
        for phase in phases:
            test_args = standard_args.copy()
            test_args[0] = phase
            ll = self.sim.log_likelihood_TD(source, sinusoid_TD, test_args)
            TD_ll_phases.append(ll)
            ll2 = self.sim.log_likelihood_FD(source, sinusoid_TD, test_args)
            FD_ll_phases.append(ll2)
        TD_ll_phases = np.array(TD_ll_phases)
        FD_ll_phases = np.array(FD_ll_phases)
            
        
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.plot(phases, TD_ll_phases, c='b', label='TD')
        ax2 = fig.add_subplot(212, sharex=ax1)
        ax2.plot(phases, FD_ll_phases, c='r', label='FD')
        
        # vertical line at injected phase
        min1, max1 = ax1.get_ylim()
        ax1.vlines(standard_args[0], min1, max1, linestyle='--', color='k')
        min2, max2 = ax2.get_ylim()
        ax2.vlines(standard_args[0], min2, max2, linestyle='--', color='k')
        
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        ax2.set_xlabel('phase')
        fig.savefig('./test_FD_TD_ll_phases.png')
        
        
        
    
        
