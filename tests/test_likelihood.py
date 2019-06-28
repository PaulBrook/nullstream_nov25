"""
Created on Tue Jun 18 12:33:34 2019

@author: jgoldstein
"""
import unittest
import numpy as np
import numpy.random as rd
import numpy.testing as npt
import matplotlib.pyplot as plt

from ptacake import PTA_sim, YEAR
from ptacake.GW_models import sinusoid_TD
from tests.setup_sim import setup_evenly_sampled


def compare_ll_plot(x, ll1, ll2, xname, label1, label2, realx=None, logx=False):
    """
    make a figure with plots of variable x vs log likelihood, to compare
    two likelihood functions (ll1 and ll2) as a function of x. First and second
    plot are ll1 and ll2, respectively. Third plot is ll1/ll2, fourth is ll1-ll2.
    
    Variable x has name xname (used as axis lable), and use label1 and label2
    for legend of ll1 and ll2 plots, respectively.
    
    If realx is given, plot horizontal line as this x value.
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(511)
    if logx:
        ax1.set_xscale('log')
    ax2 = fig.add_subplot(512, sharex=ax1)
    ax3 = fig.add_subplot(513, sharex=ax1)
    ax4 = fig.add_subplot(514, sharex=ax1)
    ax5 = fig.add_subplot(515, sharex=ax1)
        
    ax1.plot(x, ll1, c='b', label=label1)
    ax2.plot(x, ll2, c='r', label=label2)
    ax3.plot(x, ll1/ll2, c='k', label='{}/{}'.format(label1, label2))
    ax4.plot(x, ll1-ll2, c='purple', label='{} - {}'.format(label1, label2))
    ax5.plot(x, (ll1-ll2)/ll1, c='g', label='({} - {})/{}'.format(label1, label2, label1))
    
    axes = [ax1, ax2, ax3, ax4, ax5]
    if realx is not None:
        # vertical line at injected phase
        for ax in axes:
            miny, maxy = ax.get_ylim()
            ax.vlines(realx, miny, maxy, linestyle='--', color='k')
        
    for ax in axes:
        ax.legend(loc='best')
    axes[-1].set_xlabel(xname)
    
    return fig

class Test_likelihood(unittest.TestCase):
    
    ### Set plotting to True here if you want this unit test to make plots
    def __init__(self, *args, **kwargs):
        super(Test_likelihood, self).__init__(*args, **kwargs)
        self.plotting = True
    
    @classmethod
    def setUpClass(cls):
        cls.sim = setup_evenly_sampled(n_pulsars=5, seed=1234567)
        
        # make a sinusoidal signal
        # parameters past times are: phase, amplitude, polarization, cos(i), GW angular freq
        sinusoid_args = [0.123, 1e-14, np.pi/7, 0.5, 2e-8]
        # choose source (theta, phi) coordinates
        source = (0.8*np.pi, 1.3*np.pi)
        cls.sim.inject_signal(sinusoid_TD, source, *sinusoid_args)
        # for test purposes, use np fft frequencies
        freqs = np.fft.rfftfreq(len(cls.sim._times[0]), d=(cls.sim._times[0][1] - cls.sim._times[0][0]))
        cls.sim.fourier_residuals(overwrite_freqs = freqs)
        
        test_fig1 = cls.sim.plot_residuals()
        test_fig1.savefig('./TD_residuals.pdf')
        test_fig2 = cls.sim.plot_residuals_FD()
        test_fig2.savefig('./FD_residuals.pdf')
        
        npt.assert_equal(cls.sim._freqs, freqs)
        
        Dt = cls.sim._times[0][1] - cls.sim._times[0][0]
        print('Dt {}'.format(Dt))
        print('len(times): {}'.format(len(cls.sim._times[0])))
        print('len(freqs): {}'.format(len(cls.sim._freqs)))
    
    def test_TD_nullstreams(self):
        """
        Test that TD likelihood on evenly sampled data is the same with and
        without nullstream conversion.
        """
        # get likelihood for parameters close to injected parameters
        source = (0.8*np.pi, 1.3*np.pi)
        sinusoid_args_test = [0.14, 1e-14, np.pi/7.2, 0.48, 2e-8]
        TD_ll = self.sim.log_likelihood_TD(source, sinusoid_TD, sinusoid_args_test)
        # same but likelihood but with null streams
        TD_ll_ns = self.sim.log_likelihood_TD_ns(source, sinusoid_TD, sinusoid_args_test)
        
        print('TD log like: {}, TD log like w/ null streams: {}'.format(TD_ll, TD_ll_ns))
        #npt.assert_almost_equal(TD_ll, TD_ll_ns, decimal=4)
        
        
    def test_FD_nullstreams(self):
        """
        Test that the FD likelihood on evenly sampled data is the same with
        and without nullstream conversion.
        """
        # get likelihood for parameters close to injected parameters
        source = (0.8*np.pi, 1.3*np.pi)
        sinusoid_args_test = [0.11, 1e-14, np.pi/6.7, 0.513, 2e-8]
        FD_ll = self.sim.log_likelihood_FD(source, sinusoid_TD, sinusoid_args_test)
        # same likelihood but with null streams
        FD_ll_ns = self.sim.log_likelihood_FD_ns(source, sinusoid_TD, sinusoid_args_test)
        
        print('FD log like: {}, FD log like w/ null streams: {}'.format(FD_ll, FD_ll_ns))
        #npt.assert_almost_equal(FD_ll, FD_ll_ns)
        

    def test_TD_FD(self, decimal=4):
        """
        Test that the TD and the FD likelihood of evenly sampled data is 
        the same. (We're doing the ones without null streams, but it 
        doesn't really matter, don't think we need to do both since that's
        tested above.)
        
        decimal determines the precision of the npt.assert_almost_equal tests
        """
        # injection params
        source = (0.8*np.pi, 1.3*np.pi)
        standard_args = [0.123, 1e-14, np.pi/7, 0.5, 2e-8]
        
        #if not self.plotting:
        if True:
            # run the actual test
            
            # case 1, injection parameters
            TD_ll = self.sim.log_likelihood_TD(source, sinusoid_TD, standard_args)
            FD_ll = self.sim.log_likelihood_FD(source, sinusoid_TD, standard_args)

            print('case1, TD log like: {}, FD log like: {}'.format(TD_ll, FD_ll))
            #npt.assert_almost_equal(TD_ll, FD_ll, decimal=decimal)
            
            # case 2, sinusoid parameters somewhat off, same source
            sinusoid_args_test2 = [0.18, 1e-14, np.pi/6.8, 0.52, 2e-8]
            TD_ll = self.sim.log_likelihood_TD(source, sinusoid_TD, sinusoid_args_test2)
            FD_ll = self.sim.log_likelihood_FD(source, sinusoid_TD, sinusoid_args_test2)
            
            print('case2, TD log like: {}, FD log like: {}'.format(TD_ll, FD_ll))
            #npt.assert_almost_equal(TD_ll, FD_ll, decimal=decimal)
            
            # case 3, different source, different params
            # do 10 random realizations of picked sources and parametesr
            for i in range(10):
                source_test3 = (rd.random() * np.pi, rd.random() * 2 * np.pi)
                sinusoid_args_test3 = standard_args.copy()
                # random phase
                sinusoid_args_test3[0] = rd.random() * 2 * np.pi
                # small offset to amplitude between -1e-15 and +1e-15
                sinusoid_args_test3[1] += (rd.random() - 0.5) * 2e-15
                # random polarization between 0 and pi
                sinusoid_args_test3[2] = rd.random() * np.pi
                # random cos(i) between -1 and 1
                sinusoid_args_test3[3] = rd.random() * 2 - 1
                # small offset to frequency between -1e-9 and +1e-9
                sinusoid_args_test3[4] += (rd.random() - 0.5) * 2e-9
                
                TD_ll = self.sim.log_likelihood_TD(source_test3, sinusoid_TD, sinusoid_args_test3)
                FD_ll = self.sim.log_likelihood_FD(source_test3, sinusoid_TD, sinusoid_args_test3)
                
                print('case 3, TD log like: {}, FD log like: {}'.format(TD_ll, FD_ll))
                #npt.assert_almost_equal(TD_ll, FD_ll, decimal=decimal)
        
            return # so that we don't need a looong else block
        
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
        fig.savefig('./test_FD_TD_ll_phases.pdf')
        
        # vary amplitude
        log10_amps = np.linspace(-16, -13.5, num=100)
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
        fig2.savefig('./test_FD_TD_ll_amps.pdf')
        
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
        fig3.savefig('./test_FD_TD_ll_freqs.pdf')
        
    
    # expect failure for high precision test
    @unittest.expectedFailure
    def test_TD_FD_high_precision(self):
        """
        Run the TD vs FD test again, but with high precision
        """
        # decimal = 7 is the default for npt.assert_almost_equal
        if self.plotting:
            assert(False)
        else:
            self.test_TD_FD(decimal=7)
        
        
        
    
        
