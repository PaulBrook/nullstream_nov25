"""
Created on Tue Jun 18 14:50:47 2019

@author: jgoldstein
"""
import numpy as np

from .nullstream_algebra import null_streams, response_matrix


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
    
def log_likelihood_TD(self, source, model_func, model_args, **model_kwargs):
    
    # assuming evenly sampled times, same for all pulsars
    times = self._times[0]
    model_hplus, model_hcross = model_func(times, *model_args, **model_kwargs)
    responses = response_matrix(*source, self._pulsars[['theta', 'phi']].values)
    Fplus = np.expand_dims(responses[:, 0], -1)
    Fcross = np.expand_dims(responses[:, 1], -1)
    model = Fplus * model_hplus + Fcross * model_hcross

    # take inner product of residuals - model with itself
    x = self.residuals - model
    product = np.einsum('i...,ik,k...', x, self._inv_cov_residuals, x)
    ll_no_norm = -0.5 * np.sum(product)
    norm = -0.5 * len(times) * (self._n_pulsars * np.log(2 * np.pi) -np.log(np.linalg.det(self._inv_cov_residuals)))
    ll = ll_no_norm #+ norm
    
    return ll

def log_likelihood_TD_ns(self, source, model_func, model_args, **model_kwargs):#
    
    # convert residuals data to null streams
    pulsar_array = self._pulsars[['theta', 'phi']].values
    ns_data, ns_inv_cov = null_streams(self.residuals, self._inv_cov_residuals, 
                                    source, pulsar_array)
    
    # assuming evenly sampled times, same for all pulsars
    times = self._times[0]
    model_hplus, model_hcross = model_func(times, *model_args, **model_kwargs)
    
    # combine hplus, hcross with null streams to get full model
    nulls = (np.zeros_like(model_hplus),)*(self._n_pulsars-2)
    ns_model = np.hstack((model_hplus, model_hcross, *nulls)).reshape(
                                       self._n_pulsars, len(model_hplus))

    # take inner product of "null-streamed" residuals - model
    # and use null-stream version of inverse covariance in inner product
    x = ns_data - ns_model
    product = np.einsum('i...,ik,k...', x, self._inv_cov_residuals, x)
    ll_no_norm = -0.5 * np.sum(product)
    norm = -0.5 * len(times) * (self._n_pulsars * np.log(2 * np.pi) -np.log(np.linalg.det(self._inv_cov_residuals)))
    ll = ll_no_norm #+ norm
    
    return ll


### Fourier domain likelihoods ###

def log_likelihood_FD(self, source, model_func, model_args, **model_kwargs):
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
    # should already be real (since multiplying conj(x) and x)
    assert(abs(np.imag(ll_no_norm)) < abs(np.real(ll_no_norm) / 10**12))
    ll_no_norm = np.real(ll_no_norm)
        
    norm = -0.5 * len(self._freqs) * (self._n_pulsars * np.log(2 * np.pi) -np.log(np.linalg.det(self._inv_cov_residuals)))
    ll = ll_no_norm #+ norm
    
    # TODO do Parcival's theorem for unevenly sampled data and some random frequencies to
    # find what DT equivalent should be
    # lines below here work for evenly sampled data
    Dt = self._times[0][1] - self._times[0][0]
    ll_even_sampling= ll / (Dt**2 * len(self._freqs))

    # attempt at uneven sampled generalization of above: use mean of the weights instead of Dt
    all_weights = np.concatenate(self._TOA_weights)
    mean_weight = np.mean(all_weights)
    
    ll_use_mean_weight = ll / (mean_weight**2 * len(self._freqs))
    
    assert (ll_even_sampling == ll_use_mean_weight)

    return ll_use_mean_weight
    
    #fourier_mat = self._TOA_fourier_mats[0]
    #mat_product = np.dot(np.conj(fourier_mat.T), fourier_mat) 
   # return ll / (np.trace(mat_product) * len(self._freqs))
   
   
    
#    product = inner_product(x, x, self._inv_cov_residuals, self._freqs)
#    # FIXME We need a factor of 2 to make it consistent with the TD likelihood
#    # I don't know why (probably something to do with negative frequencies/ real
#    # and imaginary numbers). Possibly the inner product should be a factor 2 
#    # for TD instead of 4, and a factor 4 for FD
#    product *= 2
#    #norm = self._n_pulsars * (2*np.pi) + np.log(1/np.linalg.det(self._inv_cov_residuals))
#    return -0.5 * product #- norm


def log_likelihood_FD_ns(self, source, model_func, model_args, **model_kwargs):
    """
    Log likelihood in the Frequency Domain with null streams.
    Make sure you ran fourier_residuals beforehand!!!
    """
    # convert residuals data to null streams
    pulsar_array = self._pulsars[['theta', 'phi']].values
    ns_data, ns_inv_cov = null_streams(self.residualsFD, self._inv_cov_residuals, 
                                    source, pulsar_array)
    
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


functions = [log_likelihood_TD, log_likelihood_TD_ns, 
             log_likelihood_FD, log_likelihood_FD_ns]