#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:48:35 2019

Test to see if doing the funky fourier transforms in a separate module
(and as full block-diagonal matrices) is useful

@author: elinore
"""

import numpy as np
import scipy.linalg as la


def ift(values, freqs, times):
    """
    inverse fourier transform
    freqs: 1d array of common frequencies
    times: 2d array of desired times
    """

    # construct ift matrix
    mat = ifmat(freqs, times)

    # matrix transform returns flattened array—need to match it to times
    tvals_flat = np.real(mat @ flatten(np.array(values)))
    tvals = np.full(times.shape, np.nan)
    np.place(tvals, np.isfinite(times), tvals_flat)

    return tvals

# NOTE: compare to scipy.linalg.dft (when times are all the same)
def fmat(times, freqs):
    """
    Construct a block-diagonal matrix which gives the approximate discrete
    fourier transform between (irregular) times for each pulsar and the desired
    frequencies. To be used to convert between flattened times and freqs

    times: 2d array (Npsr x Ntimes) of times per pulsar

    frequencies: 1d array of desired frequencies
    """
    blocks = []

    # calculate the matrix for converting between each pulsar's times and freqs
    for t in times:
        t = flatten(t)
        T, F = np.meshgrid(t, freqs)
        dt = midpoint_weights(t)
        blocks += [np.exp(-2j*np.pi*F*T) * dt]

    # assemble block-diagonal complete matrix
    mat = la.block_diag(*blocks)

    return mat


def ifmat(freqs, times):
    """
    Construct a block-diagonal matrix which gives the approximate inverse
    discrete fourier transform between frequencies (potentially irregular?)
    and times (for each pulsar, irregular)

    freqs: 1d array of common frequencies

    times: 2d array of desired times

    """
    blocks = []
    df = midpoint_weights(freqs)

    # calculate each block matrix (one pulsar's times -> freqs)
    for t in times:
        t = flatten(t)
        F, T = np.meshgrid(freqs, t)
        blocks += [np.exp(2j*np.pi*F*T) * df]

    # assemble complete block-diagonal matrix
    mat = la.block_diag(*blocks)

    return mat


# blatently stolen from _PTA_sim_fourier._weights_matrix()
def midpoint_weights(seq):
    # we want the first weight to be t1 - t0, the last to be tn - t(n-1), and
    # all the rest to be 1/2 (t(i+1) - t(i)) + 1/2 (t(i) - t(i-1))
    delta_seq = np.diff(seq)
    middle_weights = 0.5 * (delta_seq[:-1] + delta_seq[1:])
    weights = np.concatenate((delta_seq[:1], middle_weights, delta_seq[-1:]))

    return weights


# routines for converting between 1d (desired for matrix multiplication)
# and 2d (clearer) representations of pulsar times
def flatten(arr):
    """
    Flatten and drop NaNs (and infs) from a 2d array of times and frequencies.
    About twice as slow as np.flatten() without dropping nans.
    """

    return arr[np.isfinite(arr)]


def expand(seq, npsr=None):
    """
    Inverse of flatten: expand a sequence to a nan-padded 2d (npsr x ntimes)
    array of increasing times. Will be somewhat more compact than random times
    generated with gaps (where longest times can have nans), but should
    otherwise match
    """

    # times should increase monotonically; reversals show a new psr starting
    # will break if one pulsar starts after the preceding one has stopped
    reversals = np.where(np.diff(seq) < 0)[0] + 1

    segments = np.split(seq, reversals)

    # check that this matches the desired number of pulsars
    if npsr is not None:
        nseg = len(segments)
        if nseg != npsr:
            raise ValueError("There are {} monotonically-increasing segments "
                             "in this sequence, so it cannot be expanded to "
                             "match {} pulsars".format(nseg, npsr))

    # nan-pad all segments to match the length of the longest one
    ntimes = np.max([len(s) for s in segments])
    expanded = [np.pad(s, (0, ntimes - len(s)), 'constant',
                       constant_values=np.nan) for s in segments]

    # combine into an (npsr x ntimes) array
    arr = np.vstack(expanded)

    return arr
