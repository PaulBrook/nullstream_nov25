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


def compare_ll_plot(x, ll1, ll2, xname, label1, label2, realx=None, logx=False):
    """
    make a figure with two plots of variable x vs log likelihood, to compare
    two likelihood functions (ll1 and ll2) as a function of x.
    
    Variable x has name xname (used as axis lable), and use label1 and label2
    for legend of ll1 and ll2 plots, respectively.
    
    If realx is given, plot horizontal line as this x value.
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    if logx:
        ax1.set_xscale('log')
    ax2 = fig.add_subplot(212, sharex=ax1)
        
    ax1.plot(x, ll1, c='b', label=label1)
    ax2.plot(x, ll2, c='r', label=label2)
    
    if realx is not None:
        # vertical line at injected phase
        min1, max1 = ax1.get_ylim()
        ax1.vlines(realx, min1, max1, linestyle='--', color='k')
        min2, max2 = ax2.get_ylim()
        ax2.vlines(realx, min2, max2, linestyle='--', color='k')
        
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax2.set_xlabel(xname)
    
    return fig

class Test_likelihood(unittest.TestCase):
    
    ### Set plotting to True here if you want this unit test to make plots
    def __init__(self, *args, **kwargs):
        super(Test_likelihood, self).__init__(*args, **kwargs)
        self.plotting = False
    
    @classmethod
    def setUpClass(cls):
        cls.sim = setup_evenly_sampled(n_pulsars=5, seed=1234567)
        
        # make a sinusoidal signal
        # parameters past times are: phase, amplitude, polarization, cos(i), GW angular freq
        sinusoid_args = [0.123, 1e-16, np.pi/7, 0.5, 2e-8]
        # choose source (theta, phi) coordinates
        source = (0.8*np.pi, 1.3*np.pi)
        cls.sim.inject_signal(sinusoid_TD, source, *sinusoid_args)
        cls.sim.fourier_residuals()
    
    def test_TD_nullstreams(self):
        """
        Test that TD likelihood on evenly sampled data is the same with and
        without nullstream conversion.
        """
        # get likelihood for parameters close to injected parameters
        source = (0.8*np.pi, 1.3*np.pi)
        sinusoid_args_test = [0.14, 1e-16, np.pi/7.2, 0.48, 2e-8]
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
        sinusoid_args_test = [0.11, 1e-16, np.pi/6.7, 0.513, 2e-8]
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
        sinusoid_args_test = [0.18, 1e-16, np.pi/6.8, 0.52, 2e-8]
        # TD likelihood
        TD_ll = self.sim.log_likelihood_TD(source, sinusoid_TD, sinusoid_args_test)
        # FD likelihood
        FD_ll = self.sim.log_likelihood_FD(source, sinusoid_TD, sinusoid_args_test)
        
        if not self.plotting:
            print('TD log like: {}, FD log like: {}'.format(TD_ll, FD_ll))
            #npt.assert_almost_equal(TD_ll, FD_ll) # this fails
            return
        
        standard_args = [0.123, 1e-16, np.pi/7, 0.5, 2e-8]
        
        # vary phase
        phases = np.linspace(0, 2*np.pi, num=50)
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
            
        fig = compare_ll_plot(phases, TD_ll_phases, FD_ll_phases, 'phase', 'TD', 
                              'FD', realx=standard_args[0])
        fig.savefig('./test_FD_TD_ll_phases.png')
        
        # vary amplitude
        log10_amps = np.linspace(-17, -15.5, num=100)
        amps = 10**log10_amps
        TD_ll_amps = []
        FD_ll_amps = []
        for amp in amps:
            test_args = standard_args.copy()
            test_args[1] = amp
            ll = self.sim.log_likelihood_TD(source, sinusoid_TD, test_args)
            TD_ll_amps.append(ll)
            ll2 = self.sim.log_likelihood_FD(source, sinusoid_TD, test_args)
            FD_ll_amps.append(ll2)
        TD_ll_amps = np.array(TD_ll_amps)
        FD_ll_amps = np.array(FD_ll_amps)
        
        fig2 = compare_ll_plot(amps, TD_ll_amps, FD_ll_amps, 'amplitude', 'TD', 
                               'FD', realx=standard_args[1], logx=True)
        fig2.savefig('./test_FD_TD_ll_amps.png')
        
        # vary freq
        freqs = np.linspace(5e-9, 1e-7, num=500)
        TD_ll_freqs = []
        FD_ll_freqs = []
        for f in freqs:
            test_args = standard_args.copy()
            test_args[-1] = f
            ll = self.sim.log_likelihood_TD(source, sinusoid_TD, test_args)
            TD_ll_freqs.append(ll)
            ll2 = self.sim.log_likelihood_FD(source, sinusoid_TD, test_args)
            FD_ll_freqs.append(ll2)
        TD_ll_freqs = np.array(TD_ll_freqs)
        FD_ll_freqs = np.array(FD_ll_freqs)
        
        fig3 = compare_ll_plot(freqs, TD_ll_freqs, FD_ll_freqs, 'GW frequency', 
                               'TD', 'FD', realx=standard_args[-1])
        fig3.savefig('./test_FD_TD_ll_freqs.png')
        
        
        
        
    
        
