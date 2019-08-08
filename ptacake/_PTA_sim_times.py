#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:48:05 2019

@author: jgoldstein

functions to go in the main PTA_sim class that have to do with setting the sampling times
"""

import numpy as np
import numpy.random as rd

try:
    from jannasutils import isIterable
except:
    # use hacked excerpt from jannasutils
    from from_jannasutils import isIterable


# all times are in seconds (or 1/seconds)
YEAR = 365.25 * 24 * 3600


# helper function to be used after setting times
def _initiate_zero_residuals(self):
    # add default (zeroed) values for other quantities
    # these are all per pulsar to be compatible with potentially different
    # numbers of TOAs per pulsar
    self._signal = np.zeros_like(self._times, dtype=float)
    self._noise = np.zeros_like(self._times, dtype=float)
    
    # values where times has a nan (padding to get rectangular arrays) are
    # also set to nan
    padding = np.isnan(self._times)
    self._signal[padding] = np.nan
    self._noise[padding] = np.nan
    
    # save the TD covariance matrix diag(sigma**2), and inverse per pulsar
    num_times = self._pulsars['nTOA'].values
    sigma2 = (self._pulsars['rms'].values)**2
    self._TD_covs = [sigma2[i] * np.eye(num_times[i]) for i in range(self._n_pulsars)]
    self._TD_inv_covs = [np.linalg.inv(cov) for cov in self._TD_covs]
    
    
def evenly_sampled_times(self, cadence=1e6, T=20*YEAR, t_start=0):
    """
    Set the same evenly sampled times for all pulsars.
    """
    times = np.arange(t_start, T, cadence)
    self._pulsars['nTOA'] = len(times)
    self._times = np.array((times,)*self._n_pulsars)
        
    # save time span for each pulsar
    self._pulsars['T'] = np.nanmax(self._times, axis=1) - np.nanmin(self._times, axis=1)

    # st residuals to zero and correct shape for times
    self._initiate_zero_residuals()
    

def randomized_times(self, mean_cadence=1e6, std_cadence=1e5,
                      min_cadence=1e5, t_start=0.0, t_end=20*YEAR,
                      gaps=True, exp_gap_spacing=5*YEAR, exp_gap_length=1e7,
                      seed=None):
    """
    Randomized times from gaussian distributed cadences.
    
    Can either have the same times for all pulsars, or be different when
    arrays are passed for the mean cadence etc per pulsar. The second option
    will give different numbers of TOAs per pulsar.
    To keep times a rectangular array, times are padded with nans.
    
    Parameters
    ----------
    mean_cadence: float or numpy array
        mean cadence for all pulsars (float) or per pulsar (array)
        default = 1e6 (seconds)
    std_cadence: float numpy array
        standard deviation for gaussian used to make randomized cadences
        default = 1e5 (seconds)
    min_cadence: float or numpy array
        minimum cadence, i.e. minimum gap between two TOAs
        default = 1e5 (seconds)
    t_start: float or numpy array
        start time
        default = 0 
    t_end: float or numpy array
        (approximate) end time
        default = 20 years
    gaps: bool
        if True, put gaps in the sampling times
        default = True
    exp_gap_spacing: float or numpy array
        expected gap spacing used to get the number of gaps from a poission distribution
        default = 5 years
    exp_gap_length: float or numpy array
        scale for the exponential distribution used to make gap lengths
        default = 1e7 seconds
    seed: None or int
        default = None
        if not None, use np.random.seed(seed) to get repeatable random behaviour
    """
    if seed is not None:
        rd.seed(seed)
    
    ### RANDOMIZED TIMES ###
    # get the number of TOAs for each pulsar
    # note: we don't take t_end to be exact, but just use it to get a number
    # of TOAs that will be randomized (so the exact start time depends on them).
    obs_time = t_end - t_start
    nTOAs = np.ceil(obs_time/ mean_cadence).astype(int)
    max_nTOAs = int(np.max(nTOAs))

    # draw cadences from a truncated gaussian distribution
    # make array rectangular by drawing the max number needed
    # this is wasteful of memory, but lets us use numpy routines more easily
    cadences = rd.normal(mean_cadence, std_cadence,
                                size=(max_nTOAs, self._n_pulsars))
    cadences = np.maximum(cadences, min_cadence)
    # had to swap shape above in rd.normal, but we want npulsars x nTOAs
    cadences = cadences.T
    
    # calculate times: start with start time, then cumulatively add cadences
    cadences[:, 0] = t_start
    times = np.cumsum(cadences, axis=1)

    # set excess times to nan: variable nTOA, but still rectangular arrays
    if isIterable(nTOAs):    
        for psr in range(self._n_pulsars):
            times[psr, nTOAs[psr]:] = np.nan
    else:
        times[:, nTOAs:] = np.nan
      
    ### GAPS ###
    if gaps:    
        # poisson dist for generating gaps (get a number of gaps for each pulsar)
        exp_ngaps = obs_time/exp_gap_spacing
        ngaps = rd.poisson(exp_ngaps, self._n_pulsars)
        
        # convert exp_gap_length to array if it's a scaler (fills array with values)
        exp_gap_length = np.broadcast_to(exp_gap_length, self._n_pulsars)

        # place gaps
        for psr in range(self._n_pulsars):
            gap_start_times = rd.random(size=ngaps[psr]) * obs_time + t_start
            gap_lengths = rd.exponential(scale=exp_gap_length[psr], size=ngaps[psr])
            
            # for each gap, set times within gap to nan
            for gap_start, gap_length in zip(gap_start_times, gap_lengths):
                in_gap = (times[psr] > gap_start) & (times[psr] < gap_start + gap_length)
                times[psr, in_gap] = np.nan

    self._times = times
    
    # save time span for each pulsar
    self._pulsars['T'] = np.nanmax(self._times, axis=1) - np.nanmin(self._times, axis=1)
    
    # save number of TOAs (not nan)
    self._pulsars['nTOA'] = [np.sum(np.isfinite(t)) for t in self._times]

    # st residuals to zero and correct shape for times
    self._initiate_zero_residuals()


def times_from_tim_file(self, filepath):
    """
    Read pulsar times from .tim file (such as from IPTA data release)
    """
    # TODO
    pass

#    # st residuals to zero and correct shape for times
#    self._initiate_zero_residuals()

functions = [_initiate_zero_residuals, evenly_sampled_times, randomized_times]
