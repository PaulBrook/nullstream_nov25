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
    self._hplus = np.zeros_like(self._times) 
    self._hcross = np.zeros_like(self._times) 
    self._signal = np.zeros_like(self._times)
    self._noise = np.zeros_like(self._times)
    
    # values where times has a nan (padding to get rectangular arrays) are
    # also set to nan
    padding = np.isnan(self._times)
    self._hplus[padding] = np.nan
    self._hcross[padding] = np.nan
    self._signal[padding] = np.nan
    self._noise[padding] = np.nan
    
def evenly_sampled_times(self, cadence=1e6, T=20*YEAR, t_start=0):
    """
    Set the same evenly sampled times for all pulsars.
    """
    times = np.arange(t_start, T, cadence)
    self._pulsars['nTOA'] = len(times)
    self._times = np.array((times,)*self._n_pulsars)

    self._initiate_zero_residuals()

def randomized_times(self, mean_cadence=1e6, std_cadence=1e5,
                      min_cadence=1e5, t_start=0.0, t_end=20*YEAR):
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

    """
    # get the number of TOAs for each pulsar
    # note: we don't take t_end to be exact, but just use it to get a number
    # of TOAs that will be randomized (so the exact start time depends on them).
    nTOAs = np.ceil((t_end - t_start)/ mean_cadence).astype(int)
    self._pulsars['nTOA'] = nTOAs
    max_nTOAs = np.max(nTOAs)

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

    self._times = times
    self._initiate_zero_residuals()


def gappy_times(self, mean_cadences=1e6, std_cadences=1e5, min_cadences=1e5,
                t_ends=20*YEAR, t_starts=0, exp_gap_spacings=5*YEAR,
                exp_gap_lengths=1e7):
    """
    Set times with random gaps and within different observation windows
    for all pulsars. All parameters can be passed as a scalar (applied to
    all pulsars equally) or an array of length n_pulsar
    """
    # get the number of TOAs for each pulsar
    nTOAs = (np.array(t_ends) - np.array(t_starts))/np.array(mean_cadences)
    nTOAs = np.ceil(nTOAs).astype(int)
    self._pulsars['nTOA'] = nTOAs

    # draw cadences from a truncated gaussian distribution
    # make array rectangular by drawing the max number needed
    # this is wasteful of memory, but lets us use numpy routines more easily
    cadences = np.random.normal(mean_cadences, std_cadences,
                                (np.max(nTOAs), self._n_pulsars))
    cadences = np.maximum(cadences, min_cadences)
    cadences[0, :] = t_starts  # initial time

    for psr in range(self._n_pulsars):
        # set excess cadences to nan: variable nTOA,
        # but still rectangular arrays so they can be treated easily
        cadences[nTOAs[psr]:, psr] = np.nan

    times = np.cumsum(cadences, 0)

    # poisson dist for generating gaps
    lambdas = (np.array(t_ends) - np.array(t_starts))/exp_gap_spacings
    ngaps = np.random.poisson(lambdas, self._n_pulsars)

    # make sure the gap lengths, mean cadences are arrays of the right size
    exp_gap_lengths = np.broadcast_to(exp_gap_lengths, self._n_pulsars)
    mean_cadences = np.broadcast_to(mean_cadences, self._n_pulsars)

    # place gaps
    for psr in range(self._n_pulsars):
        # random gap location (uniform) & length (exponential)
        gap_start_idx = np.random.randint(0, nTOAs[psr], ngaps[psr])
        gap_lengths = np.random.exponential(exp_gap_lengths[psr], ngaps[psr])

        # convert physical length to array length
        gap_points = gap_lengths/mean_cadences[psr]
        gap_points = np.floor(gap_points).astype(int)

        for g1, g2 in zip(gap_start_idx, gap_start_idx + gap_points):
            times[g1:g2, psr] = np.nan

    self._times = times.T

def times_from_tim_file(self, filepath):
    """
    Read pulsar times from .tim file (such as from IPTA data release)
    """
    # TODO
    pass

functions = [_initiate_zero_residuals, evenly_sampled_times, randomized_times, gappy_times]
