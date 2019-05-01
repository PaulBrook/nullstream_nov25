#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 17:56:53 2019

@author: jgoldstein

likelihood
only analytical phase marginalisation, rest should be done with mcmc or cpnest
always assuming single frequency data (perform likelihood evaluations per frequency)
"""

import numpy as np
from scipy.special import i0e

def likelihood_marg_phase(data, model_func, model_args, model_kwargs, inv_cov):
    """
    Likelihood at single frequency, analytically marginalise phase.
    
    Assumes the model is of the form m(parameters) * exp(i*phase), where m is
    model_func given using a phase of zero. This allows to analytically 
    marginalise over the phase here.
    
    Parameters
    ----------
    data: Numpy array
    model_func: python function
    model_params: arguments to model_func, model_func(*model_args, **model_kwargs) 
        must give a numpy array of the same shape as data
    inv_cov: inverse covariance matrix for the data
    
    Returns
    -------
    float
        likelihood
    
    """
    model = model_func(*model_args, **model_kwargs)
    
    # TODO: Should check. I forgot how this analytical phi marginalisation 
    # bessel function mess works, so I just copied it from older code.
    
    dataproduct = np.abs((np.einsum('i,ik,k', np.conj(data.T), inv_cov, data)))
    modelproduct = np.abs((np.einsum('i,ik,k', np.conj(model.T), inv_cov, model)))
    besselproduct = np.abs((np.einsum('i,ik,k', np.conj(data.T), inv_cov, model)))
        
    power = -0.5 * (dataproduct + modelproduct)
    # in some cases, exp(power) is basically 0 (numerically zero) and the bessel function is huge 
    # (numerically infinity), so we want to calculate the log directly
    # using the more well-behaved exponentially scaled bessel function i0e(x) = bessel(x) / exp(x)
    # with log(bessel(x)) = log(i0e(x)) + x
    B = np.log(i0e(besselproduct)) + besselproduct
    logl = power + B
    l = np.exp(logl)
    
    return l * 2.0*np.pi
    