#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:24:19 2019

@author: jgoldstein
"""

import numpy as np
import numpy.random as rd
import pandas as pd
from scipy.special import i0e
#import healpy as hp
#import matplotlib.pyplot as plt

try:
    from jannasutils import radec_location_to_ang, isIterable
except:
    # use hacked excerpt from jannasutils
    from from_jannasutils import radec_location_to_ang, isIterable

#from nullstream_algebra import response_matrix
from nullstream_algebra import null_streams, response_matrix
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
    
    
    def log_likelihood_TD(self, source, model_func, model_args, **model_kwargs):
        
        times = self._times[0]
        model_hplus, model_hcross = model_func(times, *model_args)
        responses = response_matrix(*source, self._pulsars[['theta', 'phi']].values)
        Fplus = np.expand_dims(responses[:, 0], -1)
        Fcross = np.expand_dims(responses[:, 1], -1)
        model = Fplus * model_hplus + Fcross * model_hcross

        x = self.residuals - model
        product_alltimes = np.einsum('i...,ik,k...', np.conj(x), self._inv_cov_residuals, x)
        real_product = 4 * np.real(product_alltimes)
        product = np.trapz(real_product, x=times)
        norm = self._n_pulsars * (2*np.pi) + np.log(1/np.linalg.det(self._inv_cov_residuals))
        print('norm', norm)
        return -0.5 * product #- norm
    
        
    def FD_inner_product(self, a, b, inv_cov):
        """
        noise-weighted inner product over frequency-domain quantities a and b.
        """
        # inner product per frequency
        product = np.einsum('i...,ik,k...', np.conj(a), inv_cov, b)
        real_product = 4 * np.real(product)
        # integrate over the frequencies
        freq_int = np.trapz(real_product, x=self._freqs)
        return freq_int
    
    
    def log_likelihood_simple(self, source, model_func, model_args, **model_kwargs):
        
        # call model function with args and preset model times, then funky fourier
        fourier_hplus, fourier_hcross = self.fourier_model(model_func, 
                                                    *model_args, **model_kwargs)
        
        # apply PTA responses to model hplus, hcross
        responses = response_matrix(*source, self._pulsars[['theta', 'phi']].values)
        Fplus = np.expand_dims(responses[:, 0], -1)
        Fcross = np.expand_dims(responses[:, 1], -1)
        model = Fplus * fourier_hplus + Fcross * fourier_hcross
        
        x = self.residualsFD - model
        product = self.FD_inner_product(x, x, self._inv_cov_residuals)
        norm = self._n_pulsars * (2*np.pi) + np.log(1/np.linalg.det(self._inv_cov_residuals))
        print('norm', norm)
        return -0.5 * product #- norm
    
    
    def log_likelihood_ns(self, source, model_func, model_args, **model_kwargs):
        """
        Log likelihood without analytical phi marginalization
        """
        # convert residuals data to null streams
        pulsar_array = self._pulsars[['theta', 'phi']].values
        ns_data, ns_inv_cov = null_streams(self.residualsFD, self._inv_cov_residuals, 
                                        source, pulsar_array)
        
        # call model function with args and preset model times, then funky fourier
        fourier_hplus, fourier_hcross = self.fourier_model(model_func, 
                                                *model_args, **model_kwargs)
        
        # combine hplus, hcross with null streams to get full model
        nulls = (np.zeros_like(fourier_hplus),)*(self._n_pulsars-2)
        ns_model = np.hstack((fourier_hplus, fourier_hcross, *nulls)).reshape(
                                           self._n_pulsars, len(fourier_hplus))
        
        x = ns_data - ns_model
        product = self.FD_inner_product(x, x, ns_inv_cov)
        norm = self._n_pulsars * (2*np.pi) + np.log(1/np.linalg.det(ns_inv_cov))
        print('norm', norm)
        return -0.5 * product #- norm
    
    
    
    
    def log_likelihood_ns_phi_marg(self, source, model_func, model_args, **model_kwargs):
        """
        Log likelihood using nullstreams, analytically marginalized over phase.
        
        Assumes the model is of the form m(parameters) * exp(i*phase), where m is
        model_func given using a phase of zero. This allows to analytically 
        marginalise over the phase here.
        Loglikelihood calculated using the fourier domain residuals. The model is
        fourier'ed inside this function, so pass a time domain model function.
        """    
        # convert residuals data to null streams
        pulsar_array = self._pulsars[['theta', 'phi']].values
        ns_data, ns_inv_cov = null_streams(self.residualsFD, self._inv_cov_residuals, 
                                        source, pulsar_array)
        
        # call model function with args and preset model times, then funky fourier
        fourier_hplus, fourier_hcross = self.fourier_model(model_func, 
                                                *model_args, **model_kwargs)
        
        # combine hplus, hcross with null streams to get full model
        nulls = (np.zeros_like(fourier_hplus),)*(self._n_pulsars-2)
        ns_model = np.hstack((fourier_hplus, fourier_hcross, *nulls)).reshape(
                                           self._n_pulsars, len(fourier_hplus))
        
        
        # TODO: Should check. I forgot how this analytical phi marginalisation 
        # bessel function mess works, so I just copied it from older code.
        
#        data_product = np.abs(np.einsum('i...,ik,k...', np.conj(ns_data), ns_inv_cov, ns_data))
#        model_product = np.abs(np.einsum('i,ik,k', np.conj(ns_model), ns_inv_cov, ns_model))
#        bessel_product = np.abs(np.einsum('i,ik,k', np.conj(ns_data), ns_inv_cov, ns_model))
        
        data_product = self.FD_inner_product(ns_data, ns_data, ns_inv_cov)
        model_product = self.FD_inner_product(ns_model, ns_model, ns_inv_cov)
        bessel_product = self.FD_inner_product(ns_data, ns_model, ns_inv_cov)
            
        power = -0.5 * (data_product + model_product)
        # in some cases, exp(power) is basically 0 (numerically zero) and the bessel function is huge 
        # (numerically infinity), so we want to calculate the log directly
        # using the more well-behaved exponentially scaled bessel function i0e(x) = bessel(x) / exp(x)
        # with log(bessel(x)) = log(i0e(x)) + x
        B = np.log(i0e(bessel_product)) + bessel_product
        logl = power + B
        #l = np.exp(logl)
        
        #return l * 2.0*np.pi
        return logl


if __name__ == '__main__':
    pass
    
    
    
