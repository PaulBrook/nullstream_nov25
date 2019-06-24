#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 16:43:50 2019

@author: elinore
"""

import numpy as np
import pandas as pd
import healpy as hp

# from .PTA_simulation import YEAR???
YEAR = 3600*24*365.25


#from .harmonics import syn_cmplx_map, gw_Cl
from harmonics import syn_cmplx_map, gw_Cl

class SkyMap:
    def __init__(self):

        self.nside = 32
        self._freqs = np.nan  # common set of frequencies for injections
        # common times for injections/pretty plots *OR* 2d uneven pulsar times
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
    def freq_map(self):
        return (self._sgwbFD + self._ephemFD + self._clockFD
                + self._indFD + self._miscFD)

    @property
    def time_map(self):
        # if they don't exist, should set up times, iMFT freq maps, and save
        return (self._sgwbTD + self._ephemTD + self._clockTD
                + self._indTD + self._miscTD)

    def get_residuals(self, psrs, times):
        # - match thetas, phis in psrs to pixels
        # - get FD residuals at those pixels
        # - evaluate iMFT of these residuals
        # - return residuals as an (Npsr x Ntimes) array
        # should accept a 2d array of times (one array for each psr)
        pass

    def MFT(self, tmap, freqs):
        # perform a matrix fourier transform
        pass

    def iMFT(self, fmap, times):
        # perform an inverse matrix fourier transform
        # times is potentially uneven

        if times.ndim == 1:
            # all pulsars have the same times, yay!
            t, f = np.meshgrid(times, self._freqs)

            # FIXME: assuming evenly spaced frequencies here
            df = np.diff(self._freqs)[0]
            ftmat = np.exp(2j*np.pi*f*t)*df
            tmap = pd.DataFrame(np.real(fmap @ ftmat), columns=times)

            return tmap
        else:
            raise NotImplementedError('Can only use a single set of times for now')


    def PSD(self, amplitude=1e-15, index=-13/3):
        # FIXME: normalization
        Sh = amplitude**2 / (12 * np.pi**2) * (self._freqs*YEAR)**index
        Sh *= YEAR**3
        Sh = pd.Series(Sh, index=self._freqs)

        return Sh

    def inject_sGWB(self, amplitude=1e-15, index=-13/3,
                    fmin=3e-9, fmax=1e-7, df=1e-9):
        """
        Generate frequency-spectrum maps of the sGWB
        """

        # if not yet defined, set up frequency bins
        if np.all(np.isnan(self._freqs)):
            self._freqs = np.arange(fmin, fmax + df, df)

        spec = pd.DataFrame(columns=self._freqs)
        spec_amp = self.PSD(amplitude=amplitude, index=index) / df

        for f in self._freqs:
            spec[f] = gw_Cl() * spec_amp[f]

        self._sgwbFD = spec.apply(syn_cmplx_map)