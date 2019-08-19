#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:16:25 2019

@author: elinore
"""

import numpy as np
import scipy

from .matrix_fourier import fmat
from .coupling import new_Y_basis
from .PTA_simulation import YEAR
from .coupling import cov_gwb_1fbin


def permute_residuals(npsr, nfreq):
    """"
    Construct a permutation matrix that will reorder timing residuals so that
    they are ordered by frequency rather than by pulsar. NOTE: all pulsars
    must have the same number of frequencies

    Parameters
    ----------
    npsr: int
        number of pulsars

    nfreq: int
        number of frequencies

    Returns
    -------
    P: array of size (npsr*nfreq) x (npsr*nfreq)
        permutation matrix to apply to residuals, eg P @ rf
    """

    # pulsar and frequency indices
    p, f = np.meshgrid(np.arange(npsr), np.arange(nfreq))

    # initial and final locations for each value
    # nb these are 2d arrays, but it doesn't seem to be a problem for indexing
    start = nfreq*p + f
    end = npsr*f + p

    # matrix should be 1 at index (desired, current) and zero elsewhere
    P = np.zeros((npsr*nfreq, npsr*nfreq))
    P[end, start] = 1

    return P


def transformation_matrix(psrs, times, freqs, weights=None,
                          drop_monopole_dipole=True, lmax=2):
    """
    Matrix to transform time-domain residuals into frequency-domain orthogonalized
    harmonic modes. Used for transforming residuals & noise covariance matrix
    """
    # Y @ P @ fmat

    npsr = len(psrs)
    nfreq = len(freqs)

    # matrix fourier transform
    F = fmat(times, freqs)

    # permute to be in frequency-order
    P = permute_residuals(npsr, nfreq)

    # orthogonalized spherical harmonics (block-diagonal matrix)
    # FIXME: weights?
    Ynew = new_Y_basis(psrs, lmax=lmax, weights=weights,
                       drop_monopole_dipole=drop_monopole_dipole)
    Ylist = [Ynew.T.values] * nfreq
    Y = scipy.linalg.block_diag(Ylist)

    # total transformation is the product of all of these
    T = Y @ P @ F

    return T


# FIXME: should make a fast version of this for likelihood.
# Drop the ephemeris/clock errors?
def cov_sGWB(Agw, freqs, index, psr, Aeph=0, Aclk=0, weights=None,
             lmax=2, drop_monopole_dipole=True):
    """
    GW covariance matrix for the frequency-domain orthogonalized harmonic modes.
    """

    Sh = Agw**2 / (12 * np.pi**2) * (freqs*YEAR)**index
    Sh *= YEAR**3

    # covariance bin for each frequency.  Assumes frequencies are independent
    # and have the same covariance matrix up to an amplitude
    covs = []
    for A in Sh:
        # FIXME: what should A actually be here?
        covs += [cov_gwb_1fbin(psr=psr, Agw=A, Aeph=Aeph, Aclk=Aclk,
                               weights=weights, lmax=lmax,
                               drop_monopole_dipole=drop_monopole_dipole)]

    cov = scipy.linalg.block_diag(covs)
    return cov


