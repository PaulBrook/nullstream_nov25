"""
Created on Tue Jun 18 14:50:47 2019

@author: jgoldstein

likelihoods for Time Domain (with and without null-streaming), and Frequency Domain
without null-streaming. FD + ns is in the _PTA_sim_nullstream module because it 
requires some more steps (concatenating).
"""
import numpy as np

from .nullstream_algebra import null_streams, response_matrix

hl2p = 0.5 * np.log(2*np.pi)

    
def log_likelihood_TD_es(self, source, model_func, model_args, 
                         add_norm=True, return_only_norm=False, **model_kwargs):
    """
    Time domain log likelihood, only for evenly sampled data.
    """
    
    # assuming evenly sampled times, same for all pulsars
    times = self._times[0]
    model_hplus, model_hcross = model_func(times, *model_args, **model_kwargs)
    responses = response_matrix(*source, self._pulsars[['theta', 'phi']].values)
    Fplus = np.expand_dims(responses[:, 0], -1)
    Fcross = np.expand_dims(responses[:, 1], -1)
    model = Fplus * model_hplus + Fcross * model_hcross
    
    # take inner product of residuals - model with itself
    x = self.residuals - model
    inv_cov = np.diag(self._pulsars['rms'].values**(-2))
    product = np.einsum('i...,ik,k...', x, inv_cov, x)
    ll = -0.5 * np.sum(product) # sum over times
    
    # use log(det(cov)) = -log(det(inv_cov))
    sign, logdet = np.linalg.slogdet(inv_cov)
    norm = len(times) * (-self._n_pulsars*hl2p + 0.5*logdet)
    
    if return_only_norm:
        return norm
    
    if add_norm:
        ll += norm
        
    return ll

# TODO: is this correct?!?!
def compute_SNR(self):
    """
    Compute signal to noise ratio of injected signal, assuming
    noise level specified by pulsar rms values. Computation in the 
    time domain.
    
    Returns
    -------
    float:
        Signal to noise ratio of injected signal
    """
    snr2 = 0
    for p in range(self._n_pulsars):
        # get mask for which times are not nan, and select non-nan signal points
        mask = np.isfinite(self._times[p])
        signal = self._signal[p][mask]
        
        # compute inner product of signal with itself, wrt noise covariance
        inv_cov = self._TD_inv_covs[p]
        product = np.einsum('i,ij,j', signal, inv_cov, signal)
        snr2 += product
        
    return np.sqrt(snr2)
    

def log_likelihood_TD(self, source, model_func, model_args, 
                      add_norm=False, return_only_norm=False, **model_kwargs):
    """
    Time domain log likelihood that works for any sampling of data.
    """
    # Because the number of times (that aren't nan) for each pulsar can be different,
    # we will need to do some part of the computation per perulsar (for-loop below)
    # but we can precompute the pulsar responses
    responses = response_matrix(*source, self._pulsars[['theta', 'phi']].values)
    Fplus = responses[:, 0]
    Fcross = responses[:, 1]
    # and we can also precompute the model, using the times that include nans
    model_hplus, model_hcross = model_func(self._times, *model_args, **model_kwargs)
    
    # compute logl and norm by summing contributions per pulsar
    ll = 0
    norm = 0
    
    for p in range(self._n_pulsars):
        # get mask for which times are not nan
        mask = np.isfinite(self._times[p])
        num_times = np.sum(mask)
        # select model for this pulsar, and use mask to select points, apply responses
        model = Fplus[p] * model_hplus[p][mask] + Fcross[p] * model_hcross[p][mask]
        
        # compute product of (res - model) * inv_covariance * (res - model)
        x = self.residuals[p][mask] - model
        inv_cov = self._TD_inv_covs[p]
        product = np.einsum('i,ij,j', x, inv_cov, x)
        ll += -0.5 * np.sum(product) # np.sum sums over times
        
        # use log(det(cov)) = -log(det(inv_cov))
        sign, logdet_inv_cov = np.linalg.slogdet(inv_cov)
        logdet_cov = - logdet_inv_cov
        #print('log( det( cov)): {}'.format(logdet_cov))
        norm += -num_times*hl2p - 0.5*logdet_cov
    
    if return_only_norm:
        return norm
    
    if add_norm:
        ll += norm
        
    return ll
    

def log_likelihood_TD_ns(self, source, model_func, model_args, 
                         add_norm=True, return_only_norm=False, **model_kwargs):
    """
    Time domain null-stream likelihood only possible for evenly sampled times.
    """
    
    # convert residuals data to null streams
    # transform inverse covariance as well
    pulsar_array = self._pulsars[['theta', 'phi']].values
    inv_cov = np.diag(self._pulsars['rms'].values**(-2))
    ns_data, ns_inv_cov = null_streams(self.residuals, inv_cov, 
                                       source, pulsar_array)
    
    # assuming evenly sampled times, same for all pulsars
    times = self._times[0]
    model_hplus, model_hcross = model_func(times, *model_args, **model_kwargs)
    
    # combine hplus, hcross with null streams to get full model
    nulls = (np.zeros_like(model_hplus),)*(self._n_pulsars-2)
    ns_model = np.hstack((model_hplus, model_hcross, *nulls)).reshape(
                                       self._n_pulsars, len(model_hplus))

    # take product of "null-streamed" residuals - model
    x = ns_data - ns_model
    product = np.einsum('i...,ik,k...', x, ns_inv_cov, x)
    ll = -0.5 * np.sum(product) # sum over times
    
    # use log(det(cov)) = -log(det(inv_cov))
    sign, logdet = np.linalg.slogdet(ns_inv_cov)
    norm = len(times) * (-self._n_pulsars*hl2p + 0.5*logdet)

    if return_only_norm:
        return norm    
    
    if add_norm:
        ll += norm
    
    return ll

   
def log_likelihood_FD(self, source, model_func, model_args, 
                      add_norm=True, return_only_norm=False, **model_kwargs):
    # call model function with args and preset model times, then funky fourier
    fourier_hplus, fourier_hcross = self.fourier_model(model_func, 
                                                *model_args, **model_kwargs)
    
    # apply response functions
    responses = response_matrix(*source, self._pulsars[['theta', 'phi']].values)
    Fplus = np.expand_dims(responses[:, 0], -1)
    Fcross = np.expand_dims(responses[:, 1], -1)
    # Npulsars x Nfreqs
    model = Fplus * fourier_hplus + Fcross * fourier_hcross
    
    # compute logl and norm by summing contributions per pulsar
    ll = 0
    norm = 0
    
    for p in range(self._n_pulsars):
        # compute product of (res - model) * inv_covariance * (res - model)
        x = self.residualsFD[p] - model[p]
        inv_cov = self._TOA_FD_inv_covs[p]
        product = np.einsum('a,ab,b', x, inv_cov, np.conj(x))
        # we think there should be a factor of 2 here to compensate for missing negative frequencies
        ll += -0.5 * 2 * np.real(np.sum(product)) # np.sum sums over frequencies
        
        # use log(det(cov)) = -log(det(inv_cov))
        sign, logdet = np.linalg.slogdet(inv_cov)
        norm += -self._n_freqs*hl2p + 0.5*logdet
        
    if return_only_norm:
        return norm        
    
    if add_norm:
        ll += norm

    return ll


functions = [log_likelihood_TD, log_likelihood_TD_ns, log_likelihood_TD_es,
             log_likelihood_FD, compute_SNR]