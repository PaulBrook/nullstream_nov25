"""
Created on Mon May 20 17:22:15 2019

@author: jgoldstein

functions to go in the main PTA_sim class that have to do with funky fouriers
"""

import numpy as np
import matplotlib.pyplot as plt


# we can make different functions like these to get different frequency spacings etc

def _weights_matrix(self, times, freqs):
    """
    Note: passing frequencies as times, and times as freqs will yield the weights
    matrix for the inverse fourier (I think?)
    """
    
    # we want the first weight to be t1 - t0, the last to be tn - t(n-1), and 
    # all the rest to be 1/2 (t(i+1) - t(i)) + 1/2 (t(i) - t(i-1))
    middle_weights = 0.5 * np.diff(times)[:-1] + 0.5 * np.diff(times)[1:]
    first_weight = np.array([times[1] - times[0]])
    last_weight = np.array([times[-1] - times[-2]])
    weights = np.concatenate((first_weight, middle_weights, last_weight))
        
    #matrix m_ij of weights(t_i) * exp(-2*pi*i*f_j*t_i)
    expanded_freqs = np.expand_dims(freqs, axis=-1)
    mat = weights * np.exp(-2*np.pi*(1j)*expanded_freqs*times)
    return weights, mat

def _setup_TOAs_fourier(self, fmax=1e-7, alpha=1, overwrite_freqs=None):
    """
    Set up for the funky fourier on the TOA residuals with unevenly sampled times.
    
    Chooses a regular grid of frequencies with fmin = 1/(alpha * Tmax), where
    Tmax is the maximum time between any two TOAs in the data set. The frequency
    spacing is also fmin. 
    Compute the weights as w(t_j)  = (1/2) (t_j+1 - t_j) + (1/2) (t_j - t_j-1), 
    with the first w_0 = t_1 - t_0, and the last weight w_n = t_n - t_n-1. 
    Then the fourier matrix is m_jk = w(t_j) * exp(-2*pi*i*f_k*t_j), with j 
    TOAs and k frequencies.
    
    Use overwrite_freqs = np.array to manually choose an array of
    frequencies instead (no guarantee they make sense)
    """
    if overwrite_freqs is not None:
        self._freqs = overwrite_freqs
    else:
        # set minimum frequency based on longest time span between any TOAs
        # with alpha some integer constant of order 1 or experiment with higher alpha
        Tmax = np.nanmax(self._times) - np.nanmin(self._times)
        fmin = 1 / (alpha * Tmax)
        # fmax chosen based on astrophysical prior, step same as fmin
        self._freqs = np.arange(fmin, fmax+fmin, step=fmin)
    self._n_freqs = len(self._freqs)
    
    # create empty fourier domain signal and noise of the right shape
    self._signalFD = np.zeros((self._n_pulsars, len(self._freqs)), dtype=np.complex_)
    self._noiseFD = np.zeros((self._n_pulsars, len(self._freqs)), dtype=np.complex_)
    
    # precomputables per pulsar: weights, fourier matrix, 
    # fourier covariance matrix, inverse and determinant
    self._TOA_weights = []
    self._TOA_fourier_mats = []
    self._TOA_FD_covs = []
    self._TOA_FD_inv_covs = []
    self._TOA_FD_cov_logdets = []
    for p in range(self._n_pulsars):
        
        # get times that aren't nan
        irregular_times = self._times[p][np.isfinite(self._times[p])]
        weights, mat = self._weights_matrix(irregular_times, self._freqs)
        self._TOA_weights.append(weights)
        self._TOA_fourier_mats.append(mat)
        FD_cov = abs(np.einsum('aj,jk,bk', mat, self._TD_covs[p], np.conj(mat)))
        self._TOA_FD_covs.append(FD_cov)
        self._TOA_FD_inv_covs.append(np.linalg.inv(FD_cov))
        sign, logdet = np.linalg.slogdet(FD_cov)
        self._TOA_FD_cov_logdets.append(logdet)
        
    
    # we have everything set up to fourier the TOA residuals now
    self._TOA_fourier_ready = True
    
def _setup_model_fourier(self, oversample=10):
    """
    Set up for the funky fourier on the model, with same frequencies as the TOA residuals.
    """
    # we need the frequencies from the TOA residuals fourier setup
    # they get initiated as None, so check for that
    if self._freqs is None:
        self._setup_TOAs_fourier()
        
    # pick a set of times to sample the model at. 
    # The maximum frequency gives us a cadence
    step = 1 / (2 * np.max(self._freqs))
    # we then oversample by some factor
    step = step / oversample
    # start before the first TOA and end after the last
    t_min = np.nanmin(self._times) - step
    t_max = np.nanmax(self._times) + step
    self._model_times = np.arange(t_min, t_max, step=step)
    
    # precomputables: weights, fourier matrix
    weights, mat = self._weights_matrix(self._model_times, self._freqs)
    self._model_weights = weights
    self._model_fourier_mat = mat
    
    self._model_fourier_ready = True

 # to do overwrite_frequencies
def fourier_residuals(self, overwrite_freqs=None):
    """
    Compute the Fourier domain signal and noise.
    
    Use overwrite_freqs = np.array to manually choose an array of
    frequencies (without guarantee they make sense). Per default the frequencies
    are chosen automatically based on the total time of the observation and 
    a reasonable maximum frequency.
    """
    # we perhaps do not need to set up every time, as long as the choice of 
    # frequencies and the time stamps of the residuals stay the same
    # but we don't expect to run this very often (just once to get the FD 
    # residuals ready). TL;DR just set up every time so it always works.
    self._setup_TOAs_fourier(overwrite_freqs=overwrite_freqs)

    for i in range(self._n_pulsars):
        
        fourier_M = self._TOA_fourier_mats[i]

        # get signal without nans
        irregularly_sampled_signal = self._signal[i][np.isfinite(self._times[i])]
        self._signalFD[i] = np.dot(fourier_M, irregularly_sampled_signal)

        if np.shape(self._noise) is not ():
            irregularly_sampled_noise = self._noise[i][np.isfinite(self._times[i])]
            self._noiseFD[i] = np.dot(fourier_M, irregularly_sampled_noise)
    
def fft_residuals(self):
    """
    Only use for evenly sample data!
    """
    ntimes = len(self._times[0])
    dt = self._times[0][1] - self._times[0][0]
    fft_freqs = np.fft.rfftfreq(ntimes, d=dt)
    self._setup_TOAs_fourier(overwrite_freqs=fft_freqs)
    
    for i in range(self._n_pulsars):
        
        self._signalFD[i] = np.fft.rfft(self._signal[i]) * dt
        
        if np.shape(self._noise) is not ():
            self._noiseFD[i] = np.fft.rfft(self._noise[i]) * dt
            
def fourier_model(self, model, *args, **kwargs):
    """
    Get funky fourier of the model (at previously set times).
    
    If you want to use custom frequency, run _setup_TOAs_fourier with
    overwrite_freqs prior to this function.
    
    Parameters
    ----------
    model: tuple of arrays (hplus, hcross) or callable
        hplus, hcross must be sampled at previously set times (PTA_sim._model_times)
        if callable, model(PTA_sim._model_times, *args, **kwargs) must return
        a tuble of (hplus, hcross) like above
    *args, **kwargs
        passed on to model if callable
    
    Returns
    -------
    tuple of numpy arrays
        fourier domain (hplus, hcross), at frequencies in PTA_sim._freqs
    """
    if not self._model_fourier_ready:
        print("First time fourier of model, setting up...")
        self._setup_model_fourier()

    if hasattr(model, '__call__'):
        hplus, hcross = model(self._model_times, *args, **kwargs)
    else:
        hplus, hcross = model
        assert hplus.shape == self._model_times.shape
        assert hcross.shape == self._model_times.shape
    
    fourier_hplus = np.dot(self._model_fourier_mat, hplus)
    fourier_hcross = np.dot(self._model_fourier_mat, hcross)
    return fourier_hplus, fourier_hcross
    

def plot_residuals_FD(self, draw_signal=True):
    """
    Plot times vs residuals for all pulsars
    """
    res = self.residualsFD
    fig, ax = plt.subplots(1)
    for i in range(self._n_pulsars):
        if draw_signal:
            ax.plot(np.log10(self._freqs), np.log10(abs(self._signalFD[i])), linewidth=0.5, alpha=0.5, c='k')
        
        ax.plot(np.log10(self._freqs), np.log10(abs(res[i])), ls='none', marker='.')
    ax.set_xlabel('log_10( frequency (Hz) )')
    ax.set_ylabel('log_10( | residualsFD (s^2) | )')
    return fig


functions = [_weights_matrix, _setup_TOAs_fourier, _setup_model_fourier, 
             fourier_residuals, fourier_model, plot_residuals_FD, fft_residuals]
