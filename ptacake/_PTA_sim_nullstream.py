"""
Created on Tue Jun 18 14:50:47 2019

@author: jgoldstein
"""
import numpy as np
import numpy.linalg as la
from scipy.linalg import block_diag
from .nullstream_algebra import construct_M
import matplotlib.pyplot as plt

    
def concatenate_residuals(self):
    """
    Concatenate FD residuals from all pulsars so we can use the null-stream stuff.
    """
    # check that fourier residuals etc have been set up
    assert self._freqs is not None, 'Null-streams only in Fourier Domain, run fourier_residuals first'
    
    self._signal_concat = self._signalFD.flatten()
    # check for noise
    if np.shape(self._noise) is not ():
        self._noise_concat = self._noiseFD.flatten()
        
    # pre-compute FD covariance for concatenated residuals
    self._big_FD_cov = block_diag(*self._TOA_FD_covs)
    self._inv_big_FD_cov = block_diag(*self._TOA_FD_inv_covs)
        

def _ns_covariance(self, small_ns_mat):
    P = self._n_pulsars
    N = self._n_freqs
    
    inv_small_ns_mat = la.inv(small_ns_mat)
    
    big_ns_mat = np.zeros((P*N, P*N))
    inv_big_ns_mat = np.zeros((P*N, P*N))
    for j in range(N):
        # starting from point [j, j], put the elements of the small ns_mat
        # at every point [j + a*N, j + b*N], with a and b any integers.
        big_ns_mat[j::N, j::N] = small_ns_mat
        inv_big_ns_mat[j::N, j::N] = inv_small_ns_mat
    
    # compute inv FD ns covariance (Eq. 23)
    Zinv = inv_big_ns_mat.T @ self._inv_big_FD_cov @ inv_big_ns_mat
    
#    # zinv by pieces: it's possible this is faster for quite large numbers of P and Nf
#    # but I couldn't pin down the scaling in some speed tests. I think it's unlikely
#    # to be faster, so let's leave it for now and perhaps come back if we find we
#    # are too slow for actual runs
#
#    Zinv = np.zeros((P*N, P*N))
#    for A, B in np.ndindex(Zinv.shape):
#        k = A//N
#        alpha = A%N
#        q = B//N
#        beta = B%N
#        
#        element = 0
#        for s, inv_sig in enumerate(self._TOA_FD_inv_covs):
#            element += inv_small_ns_mat[s, k] * inv_sig[alpha, beta] * inv_small_ns_mat[s, q]
#    
#        Zinv[A, B] = element
    
# for the normalization, we should use the covariance matrix WITHOUT the null-stream 
# so we don't need this computation below
#    # compute determinant of the ns covariance matrix
#    #(about two times faster than method below)
#    sign, log_det_ns_mat = la.slogdet(small_ns_mat)
#    log_det_Z = 2*N*log_det_ns_mat + np.sum(self._TOA_FD_cov_logdets)
#    
##    # compute deterimant from det(Zinv)
##    sign, log_det_Zinv = la.slogdet(Zinv)
##    log_det_Z = -log_det_Zinv

    return big_ns_mat, Zinv


def log_likelihood_FD_ns(self, source, model_func, model_args, 
                         add_norm=True, return_only_norm=False, **model_kwargs):
    """
    FD log likelihood using null-streams
    """
    P = self._n_pulsars
    N = self._n_freqs
    
    # get the null_stream matrix
    pulsar_array = self._pulsars[['theta', 'phi']].values
    ns_mat = construct_M(*source, pulsar_array) # the "small" ns matrix
    
    # get big (concatenated) ns matrix and ns covariance invserve
    big_ns_mat, inv_ns_cov = self._ns_covariance(ns_mat)
    
    # make null-streams out of concatenated residuals
    # @ does matrix multiplication
    null_streams = big_ns_mat @ self.residuals_concat

    # call model function with args and preset model times, then funky fourier
    fourier_hplus, fourier_hcross = self.fourier_model(model_func, *model_args, **model_kwargs)
    # combine hplus, hcross with null streams to get full model (concatenated)
    nulls = (np.zeros_like(fourier_hplus),)*(self._n_pulsars-2)
    ns_model = np.concatenate([fourier_hplus, fourier_hcross, *nulls])
    
    # compute product of data - model
    x = null_streams - ns_model
    # we think there should be a factor of 2 here to compensate for missing negative frequencies
    ll = -0.5 * 2 * (np.einsum('i,ij,j', x, inv_ns_cov, np.conj(x)))
    
    # no 0.5 in norm because complex quantity
    # for norm, use the inv_cov WITHOUT null-stream transformation
    sign, log_det_inv_cov = la.slogdet(self._inv_big_FD_cov)
    norm = N*P*np.log(2*np.pi) - log_det_inv_cov
    
    assert(abs(np.imag(ll)) < abs(np.real(ll) * 1e-10))
    ll = np.real(ll)
    
    if return_only_norm:
        return norm
    
    if add_norm:
        ll += norm
    
    return ll


functions = [concatenate_residuals, _ns_covariance, log_likelihood_FD_ns]