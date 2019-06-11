import unittest
import numpy as np
import numpy.testing as npt

from PTA_simulation import PTA_sim, YEAR


class Test_funky_fourier(unittest.TestCase):

    def test_funky_fourier1(self):
        """
        Test whether the funky fourier transform reduces to the FFT for evenly sampled
        data, and the same choice of frequencies as the FFT.
        """
        np.random.seed(1234567)
    
        ## set up PTA sim with 5 random pulsars with some varying noise levels
        sim = PTA_sim()
        sim.random_pulsars(5, sig_rms=5e-8)
        
        Dt = 1.2e6 # a bit less than 2 weeks
        T = 20*YEAR
        t_start = 0
        sim.evenly_sampled_times(cadence=Dt, T=T, t_start=t_start)
        
        #finite_times = sim._times[np.isfinite(sim._times)]
        #t_end = np.max(finite_times)
        
        ## make a test sinusoidal signal
        from GW_models import sinusoid_TD
        GW_freq = 2e-8
        GW_ang_freq = 2*np.pi*GW_freq
        # parameters past times are: phase, amplitude, polarization, cos(i), GW angular freq
        sinusoid_args = [0.123, 1e-116, np.pi/7, 0.5, GW_ang_freq]
        # choose source (theta, phi) coordinates
        source = (0.8*np.pi, 1.3*np.pi)
        sim.inject_signal(sinusoid_TD, source, *sinusoid_args)
        
        sim.plot_residuals()
        
        ## use normal fft (rfft because residuals are real) to get reference FT residuals
        n_residuals = sim._pulsars['nTOA'][0] # will be the same for all pulsars
        fft_freqs = np.fft.rfftfreq(n_residuals, d=Dt)
        # numpy implementation of fft needs to be multiplied by Delta t to get 
        # the discrete fourier transform that approximates the continues FT
        fft_residuals = np.fft.rfft(sim.residuals) * Dt
        
        ## Funky fourier with the same frequencies
        # hack build-in to setup_TOAs_fourier to manually pick the frequencies
        sim._setup_TOAs_fourier(overwrite_frequencies=fft_freqs)
        sim.fourier_residuals()
        funky_residuals = sim.residualsFD


        npt.assert_allclose(fft_residuals, funky_residuals, rtol=1e-6)
        npt.assert_allclose(fft_residuals, funky_residuals, rtol=1e-10)
        atol = np.mean(abs(fft_residuals)) / 1e10
        npt.assert_allclose(fft_residuals, funky_residuals, atol=atol)

if __name__ == '__main__':
    unittest.main()
