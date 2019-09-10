#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import numpy.testing as npt
import healpy as hp

from ptacake import PTA_sim, YEAR, SkyMap
from ptacake.matrix_fourier import fmat, flatten
#from tests.setup_sim import setup_evenly_sampled


class Test_funky_fourier(unittest.TestCase):

    def test_funky_fourier1(self):
        """
        Test whether the funky fourier transform reduces to the FFT for evenly sampled
        data, and the same choice of frequencies as the FFT.
        """
        print('running Test_funky_fourier...')

        sim = setup_evenly_sampled(seed=1234567, default_signal=True)
        sim.plot_residuals()

        Dt = sim._times[0][1] - sim._times[0][0]

        ## use normal fft (rfft because residuals are real) to get reference FT residuals
        n_residuals = sim._pulsars['nTOA'][0] # will be the same for all pulsars
        fft_freqs = np.fft.rfftfreq(n_residuals, d=Dt)
        # numpy implementation of fft needs to be multiplied by Delta t to get
        # the discrete fourier transform that approximates the continues FT
        fft_residuals = np.fft.rfft(sim.residuals) * Dt

        ## Funky fourier with the same frequencies
        # use overwrite_freqs to choose the same frequencies as the fft
        sim.fourier_residuals(overwrite_freqs=fft_freqs)
        funky_residuals = sim.residualsFD


        npt.assert_allclose(fft_residuals, funky_residuals, rtol=1e-6)
        npt.assert_allclose(fft_residuals, funky_residuals, rtol=1e-10)
        atol = np.mean(abs(fft_residuals)) / 1e10
        npt.assert_allclose(fft_residuals, funky_residuals, atol=atol)

    def test_matrix_fourier_transforms(self):
        """
        testing funky fourier transformations as implemented in the module
        matrix_fourier and the SkyMap class. Compares to numpy (i)fft routines
        and checks that a round trip returns the original results
        """

        print('running test_matrix_fourier_transformations')

        Npsr = 20

        # random stochastic GWB; freq spectrum should exactly match sim times
        sky = SkyMap(fmin=2e-9, fmax=1e-6, df=2e-9)
        sky.add_sGWB()

        # random pulsars; injection involves a matrix iFT
        sim = setup_evenly_sampled(n_pulsars=Npsr, default_signal=False,
                                   Dt=5e5, T=5e8)
        sim.inject_stochastic(sky)

        # pixels in the skymap associated with pulsars
        psr_pix = hp.ang2pix(32, sim._pulsars['theta'], sim._pulsars['phi']).values

        # standard ift of skymap values
        dt = sim._times[0, 1] - sim._times[0, 0]
        sky_fres = sky.freq_maps.loc[psr_pix].values
        zeros = np.zeros((Npsr, 1))  # add f=0 component
        sky_fres = np.concatenate((zeros, sky_fres), 1)
        ifft_sky = np.fft.irfft(sky_fres/dt)

        # set up standard fft
        nTOA = sim._pulsars['nTOA'][0]
        fft_freqs = np.fft.rfftfreq(nTOA, d=dt)
        fft_residuals = np.fft.rfft(sim.residuals) * dt

        # matrix FT:
        F = fmat(sim._times, fft_freqs)
        fres = F @ flatten(sim.residuals)
        fres = fres.reshape((Npsr, -1))

        # check that sky and sim were set up correctly
        # note: sky has no f=0 component
        npt.assert_allclose(sky.freqs, fft_freqs[1:], err_msg='Frequencies differ')

        # compare ift implementations
        # have to adjust atol b/c of residual inaccuracy (due to no f=0?)
        # suspicious: psrs have the same shift for all times (but w/sign changes)
        atol = 1e-5 * np.mean(np.abs(ifft_sky))
        npt.assert_allclose(ifft_sky, sim.residuals, atol=atol,
                            err_msg='iFFT and iMFT disagree')

        # compare forward dft's
        atol = 1e-10 * np.mean(np.abs(fres))  # much less inaccuracy
        npt.assert_allclose(fft_residuals, fres, atol=atol,
                            err_msg='FFT and MFT disagree')

        # compare round trip matrix FTs:
        atol = 1e-10 * np.mean(np.abs(fres))
        npt.assert_allclose(fres[:, :-1], sky_fres[:, :-1], atol=atol,
                            err_msg='Round-trip MFT disagrees with injection')
        # last frequency fails for some reason
        # difference is of the same order of magnitude
        # fres seems to only have real components, but sky_fres is evenly split
        npt.assert_allclose(fres[:, -1], sky_fres[:, -1], atol=atol,
                            err_msg='Round-trip MFT disagrees with injection '
                            'for the last frequency bin')



if __name__ == '__main__':
    unittest.main()
