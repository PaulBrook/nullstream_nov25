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
from .matrix_fourier import ift

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

def inject_stochastic(self, sky):
    """
    Inject stochastic correlated signal (eg GWB)

    sky: instance of SkyMap
        Frequency-domain angularly-correlated skymaps to inject
    """

    # find the frequency-domain signal for each pulsar
    try:
        pix = hp.ang2pix(sky._nside, self._pulsars.theta, self._pulsars.phi)
        f_signal = sky.freq_map.loc[pix]
    except TypeError:
        raise AttributeError('Must set up pulsars first')
    except AttributeError:
        raise AttributeError("'sky' must contain a signal")

    # inverse fourier transform
    try:
        self._signal += ift(f_signal, f_signal.columns, self._times)
    except TypeError:
        raise  AttributeError('Must set up times first')


def white_noise(self):
    """
    Inject gaussian noise according to each pulsar's rms level.
    This deletes any previously injected noise (but keeps signal the same).
    """
    # We want to get multiple output values (num times) for each value of the
    # noise rms (sigma). np.random can do this, but only if the requested
    # output size is the shape (num times, num sigma) and not the other way around.
    # So we request is in "reversed" order, then transpose the output.
    reverse_shape = self._times.T.shape
    noise = rd.normal(scale=self._pulsars['rms'].values, size=reverse_shape)
    # don't add to any previously existing noise
    self._noise = noise.T

    assert(self._noise.shape == self._times.shape)

def plot_residuals(self, draw_signal=True):
    """
    Plot times vs residuals for all pulsars
    """
    fig, ax = plt.subplots(1)
    if draw_signal:
        # plot line through signal only
        ax.plot(self._times.T, self._signal.T, ls='-', linewidth=0.5, alpha=0.7, c='k')
    ax.plot(self._times.T, self.residuals.T, ls='none', marker='.', markersize=2)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('residuals (s)')
    return fig

functions = [inject_signal, inject_stochastic, white_noise, plot_residuals]