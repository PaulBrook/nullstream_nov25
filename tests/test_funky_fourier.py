#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import numpy.testing as npt

from ptacake import PTA_sim, YEAR
from tests.setup_sim import setup_evenly_sampled


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

    def test_fmat(self):
        # fmat should be compared to np.fft.rfft(res) * dt
        # ifmat should be compared to np.fft.irfft(fres/dt)

        # evenly sampled data, skymap using the same frequencies (except zero)
        # show that a round trip preserves values:
        # skymap (f) -> residuals (t) -> fft'd residuals (f)
        pass


if __name__ == '__main__':
    unittest.main()
