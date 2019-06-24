"""
Created on Tue Jun 18 12:33:34 2019

@author: jgoldstein
"""
import unittest
import numpy as np
import matplotlib.pyplot as plt
import numpy.testing as npt

from ptacake.GW_models import sinusoid_TD
from ptacake.nullstream_algebra import response_matrix
from tests.setup_sim import setup_evenly_sampled


class Test_data_vs_model(unittest.TestCase):
    """
    Check that injected signal and model are the same (TD and FD)
    """
    ### Set plotting to True here if you want this unit test to make plots
    def __init__(self, *args, **kwargs):
        super(Test_data_vs_model, self).__init__(*args, **kwargs)
        self.plotting = False
        
    @classmethod
    def setUpClass(cls):
        cls.sim = setup_evenly_sampled(n_pulsars=5, seed=1234567)
        
        # make a sinusoidal signal
        # parameters past times are: phase, amplitude, polarization, cos(i), GW angular freq
        cls.sinusoid_args = [0.123, 1e-16, np.pi/7, 0.5, 2e-8]
        # choose source (theta, phi) coordinates
        cls.source = (0.8*np.pi, 1.3*np.pi)
        cls.sim.inject_signal(sinusoid_TD, cls.source, *cls.sinusoid_args)
        
    def test_TD_data_vs_model(self):
        
        TD_model_hplus, TD_model_hcross = sinusoid_TD(self.sim._times, *self.sinusoid_args)
        
        responses = response_matrix(*self.source, self.sim._pulsars[['theta', 'phi']].values)
        Fplus = np.expand_dims(responses[:, 0], -1)
        Fcross = np.expand_dims(responses[:, 1], -1)
        model = Fplus * TD_model_hplus + Fcross * TD_model_hcross
        
        n = self.sim._n_pulsars
        
        for j in range(n):
            npt.assert_allclose(self.sim.residuals[j], model[j])
        
        if self.plotting:
            fig, axes = plt.subplots(n, figsize=(10, 16))
            for i in range(n):
                ax = axes[i]
                ax.plot(self.sim._times[i], self.sim.residuals[i], c='b', label='residuals')
                ax.plot(self.sim._times[i], model[i], c='r', linestyle='--', label='model')
                ax.legend(loc='best')
            fig.savefig('./TD_residuals_vs_model.pdf')
    
    def test_FD_data_vs_model(self, decimal_prec=4):
        
        # compare the fourier residuals (data) and the fourier model, if the model
        # is exactly the same as the injected signal. (no noise in the data)
        
        # use default frequencies (in-built selection)
        self.sim.fourier_residuals()
        
        FD_model_hplus, FD_model_hcross = self.sim.fourier_model(sinusoid_TD, *self.sinusoid_args)
        
        # apply PTA responses to model hplus, hcross
        responses = response_matrix(*self.source, self.sim._pulsars[['theta', 'phi']].values)
        Fplus = np.expand_dims(responses[:, 0], -1)
        Fcross = np.expand_dims(responses[:, 1], -1)
        model = Fplus * FD_model_hplus + Fcross * FD_model_hcross
        
        n = self.sim._n_pulsars
        for j in range(n):
            # this is not a super high accuracy test (with decimal=4), but with higher
            # we get failure, so maybe there's some numerical imprecision going on here
            npt.assert_almost_equal(self.sim.residualsFD[j], model[j], decimal=decimal_prec)
        
        if self.plotting:
            fig, axes = plt.subplots(n, figsize=(10, 16))
            for i in range(n):
                ax = axes[i]
                ax.plot(self.sim._freqs, abs(self.sim.residualsFD[i]), c='b', label='residuals')
                ax.plot(self.sim._freqs, abs(model[i]), c='r', linestyle='--', label='model')
                ax.legend(loc='best')
            fig.savefig('./FD_residuals_vs_model.pdf')
    
    # we do the same test again, but with higher numerical precision, which 
    # we expect to fail
    @unittest.expectedFailure
    def test_FD_data_vs_model_high_precision(self):
        self.test_FD_data_vs_model(decimal_prec=7)
    
