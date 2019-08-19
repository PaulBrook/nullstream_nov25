#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 18:07:26 2019

For calculating spherical harmonic coupling matrices and new orthogonal modes
based on the coupling.

@author: elinore
"""

import numpy as np
import pandas as pd

from scipy.special import sph_harm
from .harmonics import real_Ylm, gw_Cl


# TODO: add the PTA_sim functions?

def Kllmm(psr, lmax=2, weights=None, real=True):
    """
    Coupling matrix between two spherical harmonics for a set of pulsars psr.
    Output is a (2*lmax + 1) x (2*lmax + 1) pandas dataframe showing the
    overlap between all spherical harmonics with l <= lmax for the given
    weighted pulsars.
    """

    if len(psr) < 2:
        raise ValueError('Insufficient number of pulsars to calculate coupling')

    # Build up lm dataframe
    l_repeats = np.minimum(np.arange(1, 2*(lmax + 1), 2), 2*lmax + 1)
    l_vals = np.repeat(np.arange(lmax + 1), l_repeats)
    m_vals = [np.arange(-l, l + 1) for l in range(lmax+1)]
    lm = pd.DataFrame({'m': np.concatenate(m_vals), 'l': l_vals, 'temp': 0})

    # pulsar (pixel) dataframe with weights
    pix = psr.copy()
    pix.index.name='psr'
    pix = pix.reset_index()
    pix['temp'] = 0
    if weights is not None:
        pix['w'] = weights
    else:
        pix['w'] = 1

    # get all combinations of lm, pix
    lmi = pd.merge(lm, pix, on='temp').drop('temp', axis=1)

    # spherical harmonics (function of l, m, pix)
    if real:
        # real spherical harmonic basis (useful b/c psr locations are real)
        lmi['Ylm'] = real_Ylm(lmi['l'], lmi['m'], lmi['theta'], lmi['phi'])
    else:
        # standard complex spherical harmonic basis
        lmi['Ylm'] = sph_harm(lmi['m'], lmi['l'], lmi['phi'], lmi['theta'])

    lmi = lmi.drop(['phi', 'theta'], axis=1)

    # get all combinations of different lm for each pixel
    llmmi = pd.merge(lmi.drop('w', axis=1), lmi, how='outer', on='psr')
    llmmi = llmmi.set_index(['l_x', 'm_x', 'l_y', 'm_y', 'psr'])

    # take the weighted product of spherical harmonics
    K = llmmi['w'] * llmmi['Ylm_x'] * np.conj(llmmi['Ylm_y'])

    # integrate over pixels
    norm = 4 * np.pi / pix['w'].sum()
    K = norm * K.sum(level=['l_x', 'm_x', 'l_y', 'm_y'])

    K = K.unstack(level=['l_y', 'm_y'])

    return K


# FIXME: are the weights wrong?
# new a values are given by residuals * new Y
def new_Y_basis(psr, lmax=2, drop_monopole_dipole=True, pix=None, real=True,
                weights=None, verbose=True):
    """
    Generate new basis that is orthogonal on the cut sphere defined by psr.
    Result will be a dataframe of all basis vectors built by combining
    sphereical harmonics up to lmax, evaluated in the directions given by pix,
    or at each pulsar if pix is None. Optionally, explicitly remove components
    from the monopole and dipole terms. If real is chosen, do the calculation
    using real spherical harmonics.
    """

    # coupling matrix between different m, l components
    K = Kllmm(psr, lmax=lmax, weights=weights, real=real)

    # should check if coupling matrix is well-conditioned
    if verbose:
        print('Condition number:', np.linalg.cond(K))

    # Use Gorski 1994 strategy of Cholesky decomposition for defining new basis
    L = np.linalg.cholesky(K)
    Linv = np.linalg.pinv(L)

    # set up for calculating spherical harmonics
    if pix is None:
        p = psr.copy()
    else:
        p = pix.copy()

    p['temp'] = 0
    p.index.name = 'psr'

    lm = K.index.to_frame(False)
    lm.columns = ['l', 'm']
    lm.index.name = 'lm_idx'
    lm['temp'] = 0

    lmp = pd.merge(lm.reset_index(), p.reset_index(), on='temp')
    lmp = lmp.drop('temp', 1)

    # calculate spherical harmonics at every point in p
    if real:
        lmp['Y'] = real_Ylm(lmp['l'], lmp['m'], lmp['theta'], lmp['phi'])
    else:
        lmp['Y'] = sph_harm(lmp['m'], lmp['l'], lmp['phi'], lmp['theta'])

    # index of the new orthogonal functions
    # Mas is given by the count of all all m,l <= lmax terms
    imax = (lmax + 1)**2

    # new index starts at either l=0 or l=2 (to ignore monopole, dipole terms)
    if drop_monopole_dipole:
        imin = 4
    else:
        imin = 0

    idx = np.arange(imin, imax)

    # all the Ylm's in order for each pixel in p
    pixY = lmp.set_index('lm_idx').groupby('psr')['Y']

    # Y' = L^{-1} * Y
    newY = [pixY.apply(np.dot, Linv[i,:]) for i in idx]
    newY = pd.concat(newY, axis=1, ignore_index=True)
    newY.columns = idx

    return newY

# FIXME: also need fast calculations of (C_N + C_GWB)^{-1} and det()


# noise covariance matrix will be L^† @ F @ C_N @ F^T @ L,
# where C_N is a diagonal matrix of σ^2

def cov_gwb_1fbin(psr, Agw, Aeph=0, Aclk=0, weights=None, lmax=2, real=True,
                  drop_monopole_dipole=False):
    """
    Calculate the expected covariance matrix for the new modes given a set of
    pulsars psr (with optional weights), GW power spectrum with amplitude Agw,
    ephemeris error power spectrum amplitude Aeph, clock error amplitude Aclk.
    By default calculates the contributions from quadrupole modes, but no
    higher (set by lmax). If drop_monopole_dipole is True, it returns the modes
    that are only affected by GW signal, and not those affected by monopole
    or dipole (clock or ephemeris error) effects.
    """

    # expected harmonic-space power spectrum
    # FIXME: sensible relative normalizations?
    Cl = gw_Cl(norm=Agw*6*np.pi, lmax=lmax)
    Cl[0] = Aclk
    Cl[1] = Aeph

    # expand Cl to include all the m terms (assume equal power distribution)
    l_repeats = np.arange(1, 2*(lmax+1), 2)
    Cl = np.repeat(Cl, l_repeats)
    if drop_monopole_dipole:
        Cl = Cl[4:]
    else:
        # FIXME: renormalize other l=1 modes?
        Cl[2] = 0  # no power in l=1, m=0 mode
    Cl = np.diagflat(Cl)

    # convert harmonics (Cl) to the new modes
    K = Kllmm(psr, lmax=lmax, weights=weights, real=real)
    L = np.linalg.cholesky(K)
    if drop_monopole_dipole:
        L = L[4:, 4:]

    # note: @ only works in python 3.5+, otherwise use np.matmul
    cov = np.conj(L).T @ Cl @ L

    return cov