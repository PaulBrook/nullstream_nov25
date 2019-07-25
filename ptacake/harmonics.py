#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:03:46 2019

@author: elinore
"""

import numpy as np
import pandas as pd
import healpy as hp

from scipy.special import sph_harm



def gw_Cl(norm=6*np.pi, lmax=100, df=False):
    """
    Generate power spectra satisfying the l-dependence for GWs,
    normalized by norm
    """
    Cl = np.zeros(lmax + 1)
    l = np.arange(2, lmax + 1)  # l=0 and l=1 are zero by definition

    Cl[2:] = norm/(l + 2)/(l + 1)/l/(l - 1)

    if df:
        Cl = pd.DataFrame({'Cl':Cl},
                          index=pd.Index(np.arange(lmax+1), name='l'))

    return Cl


# FIXME: generalize w/rotations
# If injected in a very narrow frequency bin, are bin effects negligeable?
def single_src_alm(lmax=10, plus=1, cross=1):
    """
    Theoretical full alms for a complex redshift map produced by a source
    situatated at the north pole. 'plus' is the relative amplitude of h+,
    'cross' is the relative amplitude of hx
    """

    l = np.arange(lmax + 1)
    l_prefactor = np.pi*(2*l + 1)/(l + 2)/(l + 1)/l/(l - 1)
    l_prefactor[:2] = 0

    h_mpos = plus + 1j*cross
    h_mneg = plus - 1j*cross
    lidx, midx = hp.Alm.getlm(lmax)

    almpos = pd.DataFrame(np.sqrt(l_prefactor)*h_mpos,
                          index=pd.MultiIndex.from_product([l, [2]]))
    almpos = almpos.reindex(pd.MultiIndex.from_arrays([lidx, midx]),
                            fill_value=0j)
    almneg = pd.DataFrame(np.sqrt(l_prefactor)*h_mneg,
                          index=pd.MultiIndex.from_product([l, [-2]]))
    almneg = almneg.reindex(pd.MultiIndex.from_arrays([lidx, -midx]),
                            fill_value=0j)

    alm = almpos.add(almneg, fill_value=0)
    alm.index.names = ['l', 'm']
    alm.columns = ['alm']

    return alm


def cmplx_map_2_full_alm(residuals, lmax):
    """
    Get the full set of alm for a complex healpix map. NaN will be treated as
    a mask. Healpy UNSEEN may be buggy for imaginary components.
    """
    l, m = hp.Alm.getlm(lmax)
    # healpix only calculates m >= 0
    pos_lm_idx = pd.MultiIndex.from_arrays((l,m), names=['l', 'm'])
    neg_lm_idx = pd.MultiIndex.from_arrays((l,-m), names=['l', 'm'])

    # mask for healpix masked array
    mask = np.isnan(residuals)
    # FIXME: add healpy UNSEEN pixels to the mask?

    # healpix assumes a real map, so split into real & imaginary parts
    real_map = hp.ma(np.real(residuals))
    real_map.mask = mask
    real_alm = hp.map2alm(real_map, lmax=lmax)

    imag_map = hp.ma(np.imag(residuals))
    imag_map.mask = mask
    imag_alm = hp.map2alm(imag_map, lmax=lmax)

    alm_real_pos = pd.Series(data=real_alm, index=pos_lm_idx)
    # al,-m = (-1)^m(al,m)*
    sign = (-1)**m
    alm_real_neg = pd.Series(data=np.conj(real_alm)*sign, index=neg_lm_idx)
    alm_real_neg.drop([0], level='m', inplace=True)
    alm_real = pd.concat([alm_real_pos, alm_real_neg])
    alm_real.sort_index(level='l', inplace=True)

    alm_imag_pos = pd.Series(data=imag_alm, index=pos_lm_idx)
    alm_imag_neg = pd.Series(data=np.conj(imag_alm)*sign, index=neg_lm_idx)
    alm_imag_neg.drop([0], level='m', inplace=True)
    alm_imag = pd.concat([alm_imag_pos, alm_imag_neg])
    alm_imag.sort_index(level='l', inplace=True)

    # spherical harmonics are a complete basis: can just add real & imaginary parts
    alm = alm_real + 1j*alm_imag

    return alm


def full_alm_2_cmplx_map(alm, nside=32, lmax=None):
    """
    Transform a full set of alm (pandas series) into a complex-valued map
    """
    l = alm.index.get_level_values('l')
    m = alm.index.get_level_values('m')

    # a_{l,-m}
    alnegm = alm.copy()
    alnegm.index = pd.MultiIndex.from_arrays([l, -m])
    alnegm = alnegm.sort_index()

    # the alm for the real and imaginary parts of the maps
    re_alm = 0.5*(alm + (-1)**np.abs(m.values) * np.conj(alnegm))
    im_alm = 0.5j*(-alm + (-1)**np.abs(m.values) * np.conj(alnegm))

    # return alms to healpy format (arrays, no neg m, sorted on m)
    re_alm = (re_alm[m >= 0]
                .sort_index(level='m')
                .values)
    im_alm = (im_alm[m >= 0]
                .sort_index(level='m')
                .values)

    re_map = hp.alm2map(re_alm, nside=nside, lmax=lmax, verbose=False)
    im_map = hp.alm2map(im_alm, nside=nside, lmax=lmax, verbose=False)

    return re_map + 1j*im_map


def full_alm_2_Cl(alm):
    """
    Get the power spectrum for a full set of alm
    """
    Cl = (alm*np.conj(alm)).sum(level='l').astype(float)
    Cl /= (2*Cl.index.values + 1)
    return Cl


def cmplx_map_2_Cl(residuals, lmax):
    """
    Wrapper for cmplx_map_2_full_alm and full_alm_2_Cl
    """
    alm = cmplx_map_2_full_alm(residuals, lmax)
    Cl = full_alm_2_Cl(alm)
    return Cl


def syn_full_alm(Cls, lmax=None):
    """
    Basically healpy synalm, but returns a sorted pandas
    series with positive and negative m (complex map).
    """
    if lmax is None:
        lmax = len(Cls) - 1

    l, m = hp.Alm.getlm(lmax)
    # index with positive, negative values of m (healpy only supplies positive)
    # (m first for ease since that's the way healpix likes it)
    mpos_idx = pd.MultiIndex.from_arrays([m, l], names=['m', 'l'])
    mneg_idx = pd.MultiIndex.from_arrays([-m, l], names=['m', 'l'])

    # complex map should be made of 2 ind. components (each is a real map)
    # Cls defined for total map, so each component should use half
    almr = pd.Series(hp.synalm(Cls*0.5, lmax, verbose=False), index=mpos_idx)
    almi = pd.Series(hp.synalm(Cls*0.5, lmax, verbose=False), index=mpos_idx)

    # combine real, imaginary components to get positive, negative m alms
    # almn relation derived from cmplx conj--neg m relation for almi, almr
    almp = almr + 1j*almi
    almn = np.conj(almr - 1j*almi)*(-1)**m
    almn.index = mneg_idx

    # positive, negative m=0 terms should match, so check & drop one
    assert np.all(almn.loc[0] == almp.loc[0]), 'Different values of alm for m = +/-0'
    almn = almn.drop(0, level='m')

    # combine & reorder to match *my* preferences
    alm = (almp.add(almn, fill_value=0j)
               .swaplevel()
               .sort_index(level='l'))

    return alm


def syn_cmplx_map(Cls, nside=32, lmax=None):
    """
    Generate a random realization of a complex skymap given a power spectrum
    Cls. Wrapper for syn_full_alm + full_alm_2_cmplx_map
    """

    alm = syn_full_alm(Cls, lmax=lmax)
    zmap = full_alm_2_cmplx_map(alm, nside=nside, lmax=lmax)

    return zmap


def real_Ylm(l, m, theta, phi):
    """
    Return the real spherical harmonics
    (standard astrophysics notation for l, m, theta, phi)
    """

    # 1 when m == 0; when m != 0, two terms are being combined
    norm = np.where(m == 0, m + 1, np.sqrt(2) * (-1)**np.abs(m))

    # complex Yl|m|
    Y_c = sph_harm(np.abs(m), l, phi, theta)

    # take real or imaginary part of Yl|m| depending on sign of m
    Y_r = norm * np.where(m < 0, np.imag(Y_c), np.real(Y_c))

    return Y_r
