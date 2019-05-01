#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:23:12 2019

@author: jgoldstein
"""
import numpy as np

def rotate_wave(Ap, Ac, psi):
    """
    Rotate the wave basis by applying rotation matrix for tensor
    
    Apply a rotation matrix based on the polarisation angle psi
    
    Parameters
    ----------
    Ap: float
        amplitude of the (unrotated) plus polarisation
    Ac: float
        amplitude of the (unrotated) cross polarisation
    psi: float
        polarisation angle in radians
        
    Returns
    -------
    float
        rotated plus polarisation amplitude
    float
        rotated cross polarisation amplitude
    """
    cos2psi = np.cos(2.0*psi)
    sin2psi = np.sin(2.0*psi)
    Ap_out = Ap*cos2psi - Ac*sin2psi
    Ac_out = Ap*sin2psi + Ac*cos2psi
    return Ap_out, Ac_out


def sinusoid_TD(times, amp, phase, pol, cosi, 
             ang_freq=2.0e-8 * np.pi, integrate=True):
    """
    Produce the GW polarisations time series according to a sinusoid model
    
    The plus and cross polarisations for a sinusoid model of a binary black hole
    (BBH) are given by:
    hplus = amp 1/2 (1+cos(i)^2) cos(2 omega t - phase)
    hcross = amp cos(i) sin(2 omega t - phase)
    with amp the overall amplitude (depending on the chirp mass for a BBH), 
    i the inclination, omega the angular orbital frequency 
    and phase the phase offset
    
    Parameters
    ----------
    times: NumPy Array
        times to calculate the GW model at
    amp: float
        Amplitude of the GW model
    phase: float
        phase offset of the GW model in radians
    pol: float
        polarisation angle in radians
    cosi: float
        cosine of the inclination, between 0 and 1
        using the cosine sice it has a flat prior
    ang_freq: float
        default: 1.0e-8 * 2pi
        angular orbital frequency of the binary
    integrate: bool
        default: True
        if True, returns time integrated gravitational waves to calculate
        residuals instead of redshifts with
    
    Returns
    -------
    NumPy Array
        Plus polarisation time series
    NumPy Array
        Cross polarisation time series
    """ 
    Aplus = amp * 0.5 * (1.0 + cosi**2.0) 
    Across = amp * cosi
    
    # GW frequency = 2 * orbital frequency
    gw_ang_freq = 2.0 * ang_freq
    
    if integrate:
        hplus = (Aplus / gw_ang_freq) * np.sin(gw_ang_freq * times + phase)
        hcross = -(Across / gw_ang_freq) * np.cos(gw_ang_freq * times + phase)
    else:
        hplus = Aplus * np.cos(gw_ang_freq * times + phase)
        hcross = Across * np.sin(gw_ang_freq * times + phase)

    # apply a rotation for the polarisation angle
    hplus, hcross = rotate_wave(hplus,hcross,pol)
    return hplus, hcross
    
def sinusoid_FD(phase, amp, pol, cosi, GW_ang_freq, Tobs, residuals=True):
    """
    Simple sinusoid model for a GW binary in frequency domain.
    
    Either residuals (default, the model is analytically integrated over time once) 
    or redshifts. 
    
    Paramters
    ---------
    phase: float
        phase offset in radians
    amp: float
        amplitude (determined by binary chirp mass, distance, and GW frequency).
    pol: float
        polarisation angle in radians
    cosi: float
        cosine of the inclination, between 0 and 1
        using the cosine sice it has a flat prior
    GW_ang_freq: float
        angular frequency of the GW signal in rad/s
        (this is twice the orbital frequency)
    Tobs: float
        duration of the GW model in the time domain, in seconds
    integrate: bool
        default: True
        if True, compute residuals from time integrated model of the redshifts
        
    Returns
    -------
    float
        plus polarisation frequency domain value at signal frequency
    float
        cross polarisation frequency domain value at signal frequency
    """
    
    Aplus = amp * 0.5 * (1.0 + cosi**2.0)
    Across = amp * cosi
    
    # fourier transformed sinusoid signal
    # 1/Tobs is the frequency bin width
    if residuals:
        hplus = Tobs * 0.5 * np.exp(1.0j * (1.5*np.pi + phase)) * (Aplus / GW_ang_freq)
        hcross = Tobs * 0.5 * np.exp(1.0j * (1.0*np.pi + phase)) * (Across / GW_ang_freq)
    else:
        hplus = Tobs * 0.5 * np.exp(1.0j * (0.0 + phase)) * Aplus
        hcross = Tobs * 0.5 * np.exp(1.0j * (1.5*np.pi + phase)) * Across
    
    # apply a rotation for the polarisation angle
    hplus, hcross = rotate_wave(hplus,hcross,pol)
    
    return hplus, hcross

def sinusoid_FD_zerophase(amp, pol, cosi, GW_ang_freq, Tobs):
    """
    Simple sinusoid model for GW binary residuals in frequency domain.
    
    To get residuals rather than redshift, the original GW model is analytically 
    integrated over time once. To use for analytical phase marginalisation, 
    the phase is always zero.
    
    Paramters
    ---------
    see sinusoid_FD (apart from phase which is set to zero)
    """
    return sinusoid_FD(0.0, amp, pol, cosi, GW_ang_freq, Tobs, residuals=True)