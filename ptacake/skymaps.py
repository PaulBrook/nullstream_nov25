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


from .harmonics import syn_cmplx_map, gw_Cl

class SkyMap:
    def __init__(self):

        self.nside = 32
        self._freqs = np.nan  # common set of frequencies
        self._times = np.nan  # common set of times

        self._sgwbFD = 0
        self._ephemFD = 0
        self._individualFD = 0

        self._sgwbTD = 0
        self._ephemTD = 0
        self._indTD = 0

    @property
    def freq_maps(self):
        return self._sgwbFD + self._ephemFD + self._clockFD + self._indFD

    @property
    def time_maps(self):
        return self._sgwbTD + self._ephemTD + self._clockTD + self._indTD

    def get_residuals(self, psrs, times):
        # - match thetas, phis in psrs to pixels
        # - get FD residuals at those pixels
        # - evaluate iMFT of these residuals
        # - return residuals as an (Npsr x Ntimes) array
        pass

    def MFT(self, freqs):
        # perform a matrix fourier transform
        pass

    def iMFT(self, times):
        # perform an inverse matrix fourier transform
        pass

    def PSD(self, amplitude=1e-15, index=-13/3):
        # FIXME: 1/2pi not generic
        Sh = amplitude**2 / (12 * np.pi**2) * (self.freqs*YEAR)**index
        Sh *= YEAR**3

        return Sh

    def inject_sGWB(self, amplitude=1e-15, index=-13/3,
                    fmin=3e-9, fmax=1e-7, df=1e-9):
        """
        Generate frequency-spectrum maps of the sGWB
        """
        self.freqs = np.arange(fmin, fmax + df, df)

        spec = pd.DataFrame(columns=self.freqs)

        for f in self.freqs:
            spec_amp = self.PSD(amplitude=amplitude, index=index)/df
            spec[f] = gw_Cl() * spec_amp

        self._sgwbFD = spec.apply(syn_cmplx_map)