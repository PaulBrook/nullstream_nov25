#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:16:25 2019

@author: elinore
"""

import numpy as np
import numpy.testing as npt
import scipy

#from .matrix_fourier import fmat, midpoint_weights
#from .coupling import new_Y_basis
#from .PTA_simulation import YEAR
#from .coupling import cov_gwb_1fbin, hellings_downs

from ptacake.matrix_fourier import fmat, midpoint_weights, permute_residuals
from ptacake.coupling import new_Y_basis
from ptacake.PTA_simulation import YEAR
from ptacake.coupling import cov_gwb_1fbin, hellings_downs

#FIXME: rewrite most of this to fit with the PTA_simulation class
# should precompute & save (as attributes) T, a, Cgw, and Cn.
# likelihood should be a method
# add monopole/dipole terms by summing Agw*Cgw + Aeph * Ceph + Aclk * Cclk?


def transformation_matrix(psrs, times, freqs, weights=None,
                          drop_monopole_dipole=True, lmax=2, verbose=False):
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
    # FIXME: weights?  Normalization seems off as well
    Ynew = new_Y_basis(psrs, lmax=lmax, weights=weights, verbose=verbose,
                       drop_monopole_dipole=drop_monopole_dipole)

    # normalization from the integral over the sky position: a = ∫ Y^† r dΩ
    if weights is None:
        Ynew *= 4*np.pi/len(psrs)
    else:
        Ynew *= 4*np.pi/np.sum(weights)
    Ylist = [Ynew.T.values] * nfreq
    Y = scipy.linalg.block_diag(*Ylist)

    # total transformation is the product of all of these
    T = Y @ P @ F

    return T


# FIXME: actually need inverse covariance matrices and determinants


# FIXME: should make a fast version of this for likelihood.
# Drop the ephemeris/clock errors and just have Agw as a separate value?
def cov_sGWB(Agw, psr, freqs, index=-13/3, Aeph=0, Aclk=0, weights=None,
             lmax=2, drop_monopole_dipole=True):
    """
    GW covariance matrix for the frequency-domain orthogonalized harmonic modes.
    """

    Sh = Agw**2 / (12 * np.pi**2) * (freqs*YEAR)**index
    Sh *= YEAR**3
    Sh /= midpoint_weights(freqs)

    # covariance bin for each frequency.  Assumes frequencies are independent
    # and have the same covariance matrix up to an amplitude
    covs = []
    for A in Sh:
        # FIXME: what should A be here? Correct normalization, powers?
        covs += [cov_gwb_1fbin(psr=psr, Agw=A, Aeph=Aeph, Aclk=Aclk,
                               weights=weights, lmax=lmax,
                               drop_monopole_dipole=drop_monopole_dipole)]

    cov = scipy.linalg.block_diag(*covs)
    return cov

# Covariance for standard HD curve


def cov_N(psr, T):
    """
    Noise covariance matrix.

    Parameters
    ----------
    psr: pandas dataframe
        pulsars. Important info is white noise rms, nTOA

    T: 2d array
        Transformation matrix from timing residuals to the new modes
    """
    # time-domain covariance matrix is a diagonal matrix of sigma^2
    # Could also use sim._TD_covs (equal to sigma2)
    sigma2 = np.repeat(psr['rms'].values**2, psr['nTOA'])
    N = np.diag(sigma2)

    # transform from TOAs to new modes
    N = T @ N @ np.conj(T.T)

    return N


def loglike(Agw, a, Cgw, Cn):
    """
    log-likelihood for a stochastic GWB (assumes monopole/dipole are removed)
    Parameters
    ----------
    Agw: float
        Amplitude of the GWB, eg 1e-15 (should probably log this)

    a: array of length (5 * Nfreq)
        harmonic frequency residuals, given by T @ r_t, where T is the
        transformation matrix

    Cgw: 2d array (5 * Nfreq) x (5 * Nfreq)
        Unnormalized stochastic GWB covariance matrix (ambiguous modes are
        dropped). Will be normalized by Agw

    Cn: 2d array (5 * Nfreq) x (5 * Nfreq)
        Noise covariance matrix
    """

    # FIXME: be smarter here?
    invcov = np.linalg.inv(Agw**2 * Cgw + Cn)

    # log determinant (should be safe from overflowing)
    sign, logdetcov = np.linalg.slogdet(Agw**2 * Cgw + Cn)
    assert sign > 0, "Determinant is not positive"

    innerproduct = a @ invcov @ a.conj()

    logl = -innerproduct - len(a)*np.log(2*np.pi) - logdetcov

    npt.assert_almost_equal(np.imag(logl), 0, err_msg='log-likelihood has a '
                            'nontrivial imaginary component')

    return np.real(logl)


def cov_gw_hd(sim, Agw, freqs, index=-13/3):
    """
    GW covariance matrix in map space, ordered by frequency
    """
    hd = hellings_downs(sim.psr_angles).values
    Sh = Agw**2 / (12 * np.pi**2) * (freqs*YEAR)**index
    Sh *= YEAR**3
    Sh /= midpoint_weights(freqs)

    covs = []
    for A in Sh:
        covs += [A*hd]

    cov = scipy.linalg.block_diag(*covs)
    return cov

