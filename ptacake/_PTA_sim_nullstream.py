"""
Created on Tue Jun 18 14:50:47 2019

@author: jgoldstein
"""
import numpy as np
import numpy.linalg as la
from scipy.linalg import block_diag
from .nullstream_algebra import construct_M
import matplotlib.pyplot as plt

l2p = np.log(2*np.pi)

    
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

    return big_ns_mat, Zinv


def log_likelihood_FD_ns(self, source, model_func, model_args, 
                         add_norm=False, return_only_norm=False, **model_kwargs):
    """
    FD log likelihood using null-streams (including reconstructed signal streams).
    Because the null-stream transformation is a linear combination of the data, 
    this log likelhood is equivalent to the "regular" FD likelihood.
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
    # a factor of 2 to compensate for the missing negative frequencies cancels out
    # the usual factor of 1/2 here
    ll = -(np.einsum('i,ij,j', x, inv_ns_cov, np.conj(x)))
    
    assert(abs(np.imag(ll)) < abs(np.real(ll) * 1e-10))
    ll = np.real(ll)

    if add_norm or return_only_norm:
        # no 0.5 in norm because complex quantity
        # for norm, use the inv_cov WITHOUT null-stream transformation
        sign, log_det_inv_cov = la.slogdet(self._inv_big_FD_cov)
        norm = - N*P*l2p + log_det_inv_cov
        if return_only_norm:
            return norm
        ll += norm
        
    return ll

def log_likelihood_FD_onlynull(self, source, add_norm=False, return_only_norm=False):
    """
    FD log-likelihood using only zero response null-streams. Because we do not 
    use the reconstructed hplus and hcross streams, this likelihood is independent 
    from the GW model; the only parameters are the source location coordinates.
    """
    P = self._n_pulsars
    N = self._n_freqs
    
    # get the null-stream matric
    pulsar_array = self._pulsars[['theta', 'phi']].values
    ns_mat = construct_M(*source, pulsar_array)
    
    # get big (concatenated) ns matrix and ns covariance inverse
    # Z = null-stream covariance
    # bigM = big null-stream matrix
    bigM, invZ = self._ns_covariance(ns_mat)
    
    # make null-streams out of concatenated residuals
    # @ does matrix multiplication
    all_null_streams = bigM @ self.residuals_concat
    
    # cut off the hplus/hcross streams (both have length N)
    null_streams = all_null_streams[2*N:]
    # and cut off the equivalent part of the invere covariance matrix
    invZ_cut = invZ[2*N:, 2*N:]
    
    # the model is all zeroes, so our usual likelihood variable x is null_streams - 0 = null_streams
    ll = -(np.einsum('i,ij,j', null_streams, invZ_cut, np.conj(null_streams)))
    
    assert(abs(np.imag(ll)) < abs(np.real(ll) * 1e-10))
    ll = np.real(ll)
    
    if add_norm or return_only_norm:
        # compute norm
        norm_part1 = -(P-2)*N*l2p
        sign, logdet_invZ_cut = la.slogdet(invZ_cut)
        logdet_Zcut = -logdet_invZ_cut
        sign, logdet_bigM = la.slogdet(bigM)
        norm_part2 = logdet_Zcut - 2*logdet_bigM
        norm = norm_part1 + norm_part2
        if return_only_norm:
            return norm
        ll += norm
        
    return ll


functions = [concatenate_residuals, _ns_covariance, log_likelihood_FD_ns, log_likelihood_FD_onlynull]