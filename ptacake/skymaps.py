#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 16:43:50 2019

@author: elinore
"""

import numpy as np
import pandas as pd
import healpy as hp
import os
import matplotlib.pyplot as plt

from .PTA_simulation import YEAR
#YEAR = 3600*24*365.25

from .harmonics import syn_cmplx_map, gw_Cl
from .matrix_fourier import midpoint_weights
#from harmonics import syn_cmplx_map, gw_Cl

class SkyMap:
    def __init__(self, fmin=1e-9, fmax=1e-7, df=1e-9, nside=32):

        self._nside = nside

        # common set of frequencies for injections
        self._freqs = np.arange(fmin, fmax + df, df)

        # common set of times for plots (or injections?)
        self._times = np.nan

        self._sgwbFD = 0
        self._ephemFD = 0
        self._clockFD = 0
        self._indFD = 0
        self._miscFD = 0

        self._sgwbTD = 0
        self._ephemTD = 0
        self._clockTD = 0
        self._indTD = 0
        self._miscTD = 0

    @property
    def freq_maps(self):
        return (self._sgwbFD + self._ephemFD + self._clockFD
                + self._indFD + self._miscFD)

    @property
    def time_maps(self):

        if np.all(np.isnan(self._times)):
            # time maps don't exist yet, so get them from the freq maps
            self.iMFT()

        return (self._sgwbTD + self._ephemTD + self._clockTD
                + self._indTD + self._miscTD)

    @property
    def freqs(self):
        return self._freqs

    @property
    def times(self):
        return self._times

    def iMFT(self, overwrite_times=None):
        """
        Transform the frequency-domain maps into time-domain maps.

        Parameters
        ----------
        overwrite_times: array-like
            Nonstandard times to use. This will overwrite any preset times
        """

        # if not yet defined, set up time bins
        if overwrite_times is not None:
            self._times = overwrite_times
        elif np.all(np.isnan(self._times)):
            T = 1/self._freqs[0]
            dt = 0.5/self._freqs[-1]
            self._times = np.arange(0, T + 0.1*dt, dt)

        t, f = np.meshgrid(self._times, self._freqs)

        df = midpoint_weights(self._freqs)
        df = df.reshape((-1, 1))
        ftmat = 2 * np.exp(2j*np.pi*f*t) * df

        # FIXME: should iFT all fields with FD injections
        self._sgwbTD = self._ift_one_field(self._sgwbFD, ftmat)
        self._clockTD = self._ift_one_field(self._clockFD, ftmat)
        self._ephemTD = self._ift_one_field(self._ephemFD, ftmat)
        self._miscTD = self._ift_one_field(self._miscFD, ftmat)


    def _ift_one_field(self, field, ftmat):
        # does the inverse fourier transform matrix multiplication
        try:
            tfield = np.real(field @ ftmat)
            tfield = pd.DataFrame(tfield, columns=self._times)
        except ValueError:
            # matrix multiplication can't handle scalars
            # but that's okay; we know this means that it's empty anyway
            tfield = 0

        return tfield


    def PSD(self, amplitude=1e-15, index=-13/3):
        """
        Generate a power-law PSD. Default is a 'normal' GWB.
        """
        Sh = amplitude**2 / (12 * np.pi**2) * (self._freqs*YEAR)**index
        Sh *= YEAR**3  # units are s^2/Hz
        Sh = pd.Series(Sh, index=self._freqs)

        return Sh


    def add_sGWB(self, amplitude=1e-15, index=-13/3):
        """
        Generate frequency-spectrum sGWB maps.
        """

        spec = pd.DataFrame(columns=self._freqs)
        df = midpoint_weights(self._freqs)
        spec_amp = self.PSD(amplitude=amplitude, index=index) / df
        #spec_amp = self.PSD(amplitude=amplitude, index=index)

        for f in self._freqs:
            spec[f] = 2 * gw_Cl() * spec_amp[f]

        # units are s/Hz
        self._sgwbFD = spec.apply(syn_cmplx_map, args=[self._nside])


    def add_correlated_signal(self, PSD, Cl):
        """
        Generate an arbitrary angularly-correlated signal

        Parameters:
        ----------
        PSD: array-like
            Frequency power spectrum. Should match the internal frequencies

        Cl: array-like
            Angular power spectrum. For clock or ephemeris errors, should
            be zero except for l=0 or l=1
        """

        if len(PSD) != len(self._freqs):
            raise ValueError('PSD must match the preset frequencies')

        spec = pd.DataFrame(columns=self._freqs)
        df = midpoint_weights(self._freqs)
        spec_amp = pd.Series(PSD/df)
        spec_amp.index = self._freqs

        for f in self._freqs:
            # this could definitely be done without the for loop
            spec[f] = np.array(Cl) * spec_amp[f]

        # check if this is an injected monopole, dipole, or something else
        l = Cl[Cl > 0]

        if np.all(l == 0):
            # monopole injection (clock error)
            self._clockFD = spec.apply(syn_cmplx_map, args=[self._nside])
        elif np.all(l == 1):
            # dipole injection (ephemeris error)
            self._ephemFD = spec.apply(syn_cmplx_map, args=[self._nside])
        else:
            # who the heck knows
            self._miscFD = spec.apply(syn_cmplx_map, args=[self._nside])


    def plot_skymaps(self, filepath, time=True, **kwargs):
        """
        Generate plots of the skymaps and save them to the directory given by
        filepath. Set time to False to generate the frequency maps. Keyword
        arguments will be passed to healpy.
        """

        if time:
            maps = self.time_maps
            labels = self._times/YEAR
            label_unit = 'years'
            default_cmap = plt.cm.RdBu_r
            default_cmap.set_under('w')
            default_vmax = np.max(np.abs(np.ravel(maps)))
            default_vmin = -default_vmax
            default_unit = 'timing residuals (s)'
        else:
            maps = np.abs(self.freq_maps)
            labels = self._freqs * 1e9
            label_unit = 'nHz'
            default_cmap = plt.cm.Reds
            default_cmap.set_under('w')
            default_vmax = None
            default_vmin = 0
            default_unit = ''

        # set up default plotting params & make user-specified adjustments
        hp_kws = dict(max=default_vmax, min=default_vmin, cmap=default_cmap,
                      format='%.2g', unit=default_unit)
        hp_kws.update(kwargs)

        for m, l in zip(maps, labels):
            hp.mollview(maps[m], title='{0:.2g} {1}'.format(l, label_unit), **hp_kws)
            filename = '{0:.2f}_{1}.pdf'.format(l, label_unit)
            plt.savefig(os.path.join(filepath, filename))
            plt.close()
