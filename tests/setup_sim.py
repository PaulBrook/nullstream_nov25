"""
Created on Mon Jun 17 14:32:20 2019

@author: jgoldstein

some setup function(s) for tests
"""
import numpy as np
import numpy.random as rd

from ptacake import PTA_sim, YEAR


def setup_evenly_sampled(n_pulsars=5, seed=None, default_signal=False,
                         Dt=2.0e6, T=15*YEAR):
    """
    Setup an example sim with evenly sampled times and a sinusoid signal, no noise.

    Note: simulation parameters such as cadence and GW params are already chosen.
    Edit the code or add keyword params if you want to change these. You can change
    the number of pulsars with the keyword param n_pulsars.

    Parameters
    ----------
    n_pulsars: int
        number of pulsars to use in sim
        default = 5
    seed: None or int
        if int, use as random seed
        default = None
    default_signal: bool
        if True, also inject some standard sinusoid signal
        default = False

    Returns
    -------
    PTA_sim
        The sim object with evenly sampled times, and an injected sinusoid signal
    """
    if seed is not None:
        rd.seed(seed)

    ## set up PTA sim with 5 random pulsars with some varying noise levels
    sim = PTA_sim()
    sim.random_pulsars(n_pulsars, sig_rms=5e-8)

    t_start = 0
    sim.evenly_sampled_times(cadence=Dt, T=T, t_start=t_start)

    #finite_times = sim._times[np.isfinite(sim._times)]
    #t_end = np.max(finite_times)

    if default_signal:
        ## make a test sinusoidal signal
        from ptacake.GW_models import sinusoid_TD
        # parameters past times are: phase, amplitude, polarization, cos(i), GW angular freq
        sinusoid_args = [0.123, 1e-14, np.pi/7, 0.5, 2e-8]
        # choose source (theta, phi) coordinates
        source = (0.8*np.pi, 1.3*np.pi)
        sim.inject_signal(sinusoid_TD, source, *sinusoid_args)

    return sim