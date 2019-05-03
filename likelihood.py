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

from nullstream_algebra import null_streams


def strain_to_null_stream_model(strain_model, model_args, model_kwargs, n_pulsars):
    """
    From hplus, hcross model, get model of hplus, hcross, null streams.
    """
    hplus, hcross = strain_model(*model_args, **model_kwargs)
    return np.hstack((hplus, hcross, (0,)*(n_pulsars-2)))

def log_like_marg_phase(data, invC, source, pulsars, model_func, model_args, model_kwargs):
    """
    Likelihood at single frequency, analytically marginalise phase.
    
    Assumes the model is of the form m(parameters) * exp(i*phase), where m is
    model_func given using a phase of zero. This allows to analytically 
    marginalise over the phase here.
    
    Parameters
    ----------
    data: numpy array
        single frequency residuals for each pulsar
    invC: inverse covariance matrix for the residuals data
    source: array-like
        (theta, phi) coordinate of the source
    pulsars: numpy array
        (theta, phi) coordinates of all pulsars
    model_func: python function
    model_args: arguments to model_func, model_func(*model_args, **model_kwargs) 
        must give a numpy array of the same shape as data
    model_kwargs: (optional) key-word arguments passed to model_func
        use empty dict (dict()) if not using keyword args

    Returns
    -------
    float
        likelihood
    
    """
    n_pulsars = len(data)
    # convert residuals data to null streams
    ns_data, inv_cov = null_streams(data, invC, source, pulsars)
    # get model point including null streams
    model = strain_to_null_stream_model(model_func, model_args, model_kwargs, n_pulsars)
    
    # TODO: Should check. I forgot how this analytical phi marginalisation 
    # bessel function mess works, so I just copied it from older code.
    
    dataproduct = np.abs((np.einsum('i,ik,k', np.conj(ns_data.T), inv_cov, ns_data)))
    modelproduct = np.abs((np.einsum('i,ik,k', np.conj(model.T), inv_cov, model)))
    besselproduct = np.abs((np.einsum('i,ik,k', np.conj(ns_data.T), inv_cov, model)))
        
    power = -0.5 * (dataproduct + modelproduct)
    # in some cases, exp(power) is basically 0 (numerically zero) and the bessel function is huge 
    # (numerically infinity), so we want to calculate the log directly
    # using the more well-behaved exponentially scaled bessel function i0e(x) = bessel(x) / exp(x)
    # with log(bessel(x)) = log(i0e(x)) + x
    B = np.log(i0e(besselproduct)) + besselproduct
    logl = power + B
    #l = np.exp(logl)
    
    #return l * 2.0*np.pi
    return logl
    