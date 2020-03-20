"""
Created on Mon May 20 17:26:51 2019

@author: jgoldstein

functions to go in the main PTA_sim class that have to do with signal/noise injections
"""
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import healpy as hp

from .nullstream_algebra import response_matrix
from .matrix_fourier import ift, ifmat, flatten


def inject_signal(self, signal_func, source, *signal_args, **signal_kwargs):
    """
    Inject signal into the residuals.
    """
    hplus, hcross = signal_func(self._times, *signal_args, **signal_kwargs)
    responses = response_matrix(*source, self._pulsars[['theta', 'phi']].values)
    Fplus = np.expand_dims(responses[:, 0], -1)
    Fcross = np.expand_dims(responses[:, 1], -1)

    # add to current signal in case we want multiple injections done
    self._signal += Fplus * hplus + Fcross * hcross


def inject_stochastic(self, sky, pulsar_term=False, random_state=None):
    """
    Inject stochastic correlated signal (eg GWB)

    sky: instance of SkyMap
        Frequency-domain angularly-correlated skymaps to inject

    pulsar_term: bool
        If true, include a random pulsar term from the GWB
    """

    # find the frequency-domain signal for each pulsar
    try:
        pix = hp.ang2pix(sky._nside, self._pulsars.theta, self._pulsars.phi)
        f_signal = sky.freq_maps.loc[pix]

    except AttributeError as err:
        if str(err) == "'int' object has no attribute 'loc'":
            raise AttributeError("'SkyMap' object does not contain a signal") from err
        else:
            raise err
    except TypeError as err:
        if '_ang2pix_ring' in str(err):
            raise AttributeError('Pulsars have not been set up') from err
        else:
            raise err

    # inverse fourier transform
    try:
        mat = ifmat(f_signal.columns, self._times)
        self._signal += ift(f_signal, f_signal.columns, self._times, mat=mat)
    except TypeError as err:
        if str(err) == "'int' object is not iterable":
            raise  AttributeError('Times have not been set up') from err
        else:
            raise err

    # add pulsar term
    if pulsar_term:
        npsr = len(pix)
        pf_shape = np.shape(f_signal)
        pf = np.zeros(pf_shape, dtype=np.complex)
        # choose random gwb amplitudes for each frequency for each pulsar
        for i, f in enumerate(sky._sgwbFD.columns):
            random_fs = sky._sgwbFD[f].sample(npsr, random_state=random_state)
            pf[:, i] = random_fs
        #random_gw_f = sky._sgwbFD.sample(len(pix))
        pterm = ift(pf, f_signal.columns, self._times, mat=mat)
        self._noise += pterm


def white_noise(self, seed=1000, scale=1):
    """
    Inject gaussian noise according to each pulsar's rms level. This deletes 
    any previously injected noise (but keeps signal the same). Use keyword 
    scale to inject white noise scaled up or down relative to pulsar rms levels, 
    for example scale=0.1 to inject 10% of noise (keeps covariance matrix the same).
    """
    # We want to get multiple output values (num times) for each value of the
    # noise rms (sigma). np.random can do this, but only if the requested
    # output size is the shape (num times, num sigma) and not the other way around.
    # So we request is in "reversed" order, then transpose the output.
    reverse_shape = self._times.T.shape
    rd.seed(seed)
    noise = rd.normal(scale=self._pulsars['rms'].values, size=reverse_shape)
    # don't add to any previously existing noise
    # scale with keyword scale
    self._noise = scale * noise.T

    assert(self._noise.shape == self._times.shape)


def plot_residuals(self, draw_signal=True, include_noise=True):
    """
    Plot times vs residuals for all pulsars

    Returns
    -------
    (fig, ax) matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(1)
    if draw_signal:
        # plot line through signal only
        ax.plot(self._times.T, self._signal.T, ls='-', linewidth=0.5, alpha=0.7, c='k')
    if include_noise:
        residuals = self.residuals
    else:
        residuals = self._signal
    ax.plot(self._times.T, residuals.T, ls='none', marker='.', markersize=2)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('residuals (s)')
    return fig, ax

functions = [inject_signal, inject_stochastic, white_noise, plot_residuals]