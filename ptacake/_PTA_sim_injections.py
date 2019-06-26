"""
Created on Mon May 20 17:26:51 2019

@author: jgoldstein

functions to go in the main PTA_sim class that have to do with signal/noise injections
"""
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

from .nullstream_algebra import response_matrix

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

def white_noise(self):
    """
    Inject gaussian noise according to each pulsar's rms level.
    This deletes any previously injected noise (but keeps signal the same).
    """
    # annoyingly, rd.normal cannot handle both an array for the scale values
    # and more than a scalar output for each of those values, so we have to
    # loop through the 
    # ---> it can, but only in the reverse order (so the scale shape is the
    # same as the last, not the first axis). So we transpose at the end.
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
        # plot line through signal only, use the same colour as the corresponding residuals
        ax.plot(self._times.T, self._signal.T, ls='-', linewidth=0.5, alpha=0.5, c='k')
    ax.plot(self._times.T, self.residuals.T, ls='none', marker='.', markersize=2)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('residuals (s)')
    return fig

functions = [inject_signal, white_noise, plot_residuals]