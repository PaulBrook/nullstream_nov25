"""
Created on Tue Jun 18 14:50:47 2019

@author: jgoldstein
"""
import numpy as np

from .nullstream_algebra import null_streams, response_matrix

hl2p = 0.5 * np.log(2*np.pi)


### Noise-weighted inner product (TD or FD) ### 

def inner_product(a, b, inv_cov, steps):
    """
    noise-weighted inner product over quantities a and b
    steps can be either times or freqs for TD or FD inner product,
    as long as a = a(steps), b = b(steps).
    """
    # inner product per step
    product = np.einsum('i...,ik,k...', np.conj(a), inv_cov, b)
    real_product = 4 * np.real(product)
    # integrate over the steps (times or freqs)
    integ = np.trapz(real_product, x=steps)
    return integ
    #return np.sum(product)


### Time domain likelihoods ####
    
def log_likelihood_TD_es(self, source, model_func, model_args, **model_kwargs):
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
    ll_no_norm = -0.5 * np.sum(product) # sum over times
    # use log(det(cov)) = -log(det(inv_cov))
    sign, logdet = np.linalg.slogdet(inv_cov)
    norm = len(times) * (-self._n_pulsars*hl2p + 0.5*logdet)
    ll = ll_no_norm #+ norm
    
    return ll#, norm

def log_likelihood_TD(self, source, model_func, model_args, **model_kwargs):
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
    logl = 0
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
        logl += -0.5 * np.sum(product) # np.sum sums over times
        # use log(det(cov)) = -log(det(inv_cov))
        sign, logdet = np.linalg.slogdet(inv_cov)
        norm += -num_times*hl2p + 0.5*logdet
    
    ll = logl #+ norm
    return ll#, norm
    

def log_likelihood_TD_ns(self, source, model_func, model_args, **model_kwargs):
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
    
    ll_no_norm = -0.5 * np.sum(product) # sum over times
    # use log(det(cov)) = -log(det(inv_cov))
    sign, logdet = np.linalg.slogdet(ns_inv_cov)
    norm = len(times) * (-self._n_pulsars*hl2p + 0.5*logdet)
    ll = ll_no_norm #+ norm
    
    return ll#, norm


### Fourier domain likelihoods ###

def log_likelihood_FD_test(self, source, model_func, model_args, **model_kwargs):
    """
    Log likelihood in the Frequency Domain (without null streams)
    Make sure you ran fourier_residuals beforehand!!!
    """  
    # call model function with args and preset model times, then funky fourier
    fourier_hplus, fourier_hcross = self.fourier_model(model_func, 
                                                *model_args, **model_kwargs)
    
    # apply PTA responses to model hplus, hcross
    responses = response_matrix(*source, self._pulsars[['theta', 'phi']].values)
    Fplus = np.expand_dims(responses[:, 0], -1)
    Fcross = np.expand_dims(responses[:, 1], -1)
    model = Fplus * fourier_hplus + Fcross * fourier_hcross
    
    # take inner product of x = (residuals - model) with itself
    x = self.residualsFD - model
    
    product = np.einsum('i...,ik,k...', np.conj(x), self._inv_cov_residuals, x)
    ll_no_norm = -0.5 * np.sum(product)
    norm = -0.5 * len(self._freqs) * (self._n_pulsars * np.log(2 * np.pi) -np.log(np.linalg.det(self._inv_cov_residuals)))
    ll = ll_no_norm #+ norm

    print('un-corrected ll {}'.format(ll))

    # TODO do Parcival's theorem for unevenly sampled data and some random frequencies to
    # find what DT equivalent should be
    # lines below here work for evenly sampled data
    Dt = self._times[0][1] - self._times[0][0]
    ll_even_sampling= ll / (Dt**2 * len(self._freqs))
    print('corrected with Dt (evenly sampled) ll {}'.format(ll_even_sampling))

    # attempt at uneven sampled generalization of above: use mean of the weights instead of Dt
    all_weights = np.concatenate(self._TOA_weights)
    mean_weight = np.mean(all_weights)
    
    ll_use_mean_weight = ll / (mean_weight**2 * len(self._freqs))
    print('corrected with mean weight ll {}'.format(ll_use_mean_weight))
    
    # this reduces to the Dt method when using evenly sampled data, because we can
    # do the following assert without failure
    #assert (ll_even_sampling == ll_use_mean_weight)
    
    # instead of using mean weight, use mean cadence
    # they are close, but not exactly the same
    # mean cadence = T/N = (1/N) * (t_n - t_1)
    # mean weight = (1/N) * [(3/2)*T - (1/2)*(t_N-1 - t_2)]
    
    # for mean cadence, divide by N-1 not N, to get the average step between times
    mean_cadence = np.mean(self._pulsars['T'] / (self._pulsars['nTOA']-1))
    ll_use_mean_cadence = ll / (mean_cadence**2 * len(self._freqs))
    
    print('corrected with mean cadence ll {}'.format(ll_use_mean_cadence))

    # this also reduces to using Dt for evenly sampled data
    #assert (ll_use_mean_weight == ll_use_mean_cadence)
    
    return ll_use_mean_cadence
    
    #fourier_mat = self._TOA_fourier_mats[0]
    #mat_product = np.dot(np.conj(fourier_mat.T), fourier_mat) 
   # return ll / (np.trace(mat_product) * len(self._freqs))
   
   
def log_likelihood_FD(self, source, model_func, model_args, **model_kwargs):
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
    logl = 0
    norm = 0
    
    for p in range(self._n_pulsars):
        # compute product of (res - model) * inv_covariance * (res - model)
        x = self.residualsFD[p] - model[p]
        inv_cov = self._TOA_FD_inv_covs[p]
        product = np.einsum('a,ab,b', x, inv_cov, np.conj(x))
        # should there be a times 2 here, to compensate for missing negative frequencies?
        logl += -0.5 * 2 * np.real(np.sum(product)) # np.sum sums over frequencies
        # use log(det(cov)) = -log(det(inv_cov))
        sign, logdet = np.linalg.slogdet(inv_cov)
        norm += -self._n_freqs*hl2p + 0.5*logdet
    
    ll = logl #+ norm
    return ll#, norm


def log_likelihood_FD_ns_(self, source, model_func, model_args, **model_kwargs):
    """
    Log likelihood in the Frequency Domain with null streams.
    Make sure you ran fourier_residuals beforehand!!!
    """
    # convert residuals data to null streams
    pulsar_array = self._pulsars[['theta', 'phi']].values
    ns_data, ns_inv_cov = null_streams(self.residualsFD, self._TOA_FD_inv_covs, source, pulsar_array)
    
    # call model function with args and preset model times, then funky fourier
    fourier_hplus, fourier_hcross = self.fourier_model(model_func, 
                                            *model_args, **model_kwargs)
    
    # combine hplus, hcross with null streams to get full model
    nulls = (np.zeros_like(fourier_hplus),)*(self._n_pulsars-2)
    ns_model = np.hstack((fourier_hplus, fourier_hcross, *nulls)).reshape(
                                       self._n_pulsars, len(fourier_hplus))
    
    # take inner product of x = (residuals - model) with itself
    x = ns_data - ns_model
    product = inner_product(x, x, ns_inv_cov, self._freqs)
    # See log_likelihood_FD comment
    product *= 2
    #norm = self._n_pulsars * (2*np.pi) + np.log(1/np.linalg.det(ns_inv_cov))
    return -0.5 * product #- norm


functions = [log_likelihood_TD, log_likelihood_TD_ns, log_likelihood_TD_es,
             log_likelihood_FD, log_likelihood_FD_ns_]