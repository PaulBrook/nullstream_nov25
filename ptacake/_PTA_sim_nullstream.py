"""
Created on Tue Jun 18 14:50:47 2019

@author: jgoldstein
"""
import numpy as np
import numpy.linalg as la
from scipy.linalg import block_diag
from .nullstream_algebra import construct_M

    
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
        

def _ns_covariance(self, small_ns_mat):
    P = self._n_pulsars
    N = self._n_freqs
    
    big_ns_mat = np.zeros((P*N, P*N))
    for j in range(N):
        # starting from point [j, j], put the elements of the small ns_mat
        # at every point [j + a*N, j + b*N], with a and b any integers.
        big_ns_mat[j::N, j::N] = small_ns_mat
    
#    # for-loop method of computing the ns FD covariance (Eq. 19)
#    Z = np.zeros((P*N, P*N))
#    for A, B in np.ndindex(Z.shape):
#        k = A//N
#        alpha = A%N
#        l = B//N
#        beta = B%N
#
#        # NOTE: I tested an alternative to the lines below by using an array of the
#        # TOA_FD_covs and no for-loop, but it is slightly slower
#        element = 0
#        for s, sig in enumerate(self._TOA_FD_covs):
#            element += small_ns_mat[k, s] * sig[alpha, beta] * small_ns_mat[l, s]
#        
#        Z[A, B] = element
        
    # big matrix multiplication method to compute Z (lot faster!)
    big_FD_cov = block_diag(*self._TOA_FD_covs)
    Z = big_ns_mat @ big_FD_cov @ np.conj(big_ns_mat.T)
    
    # compute determinant of the ns covariance matrix
    sign, log_det_ns_mat = la.slogdet(small_ns_mat)
    log_det_Z = 2*N*log_det_ns_mat + np.sum(self._TOA_FD_cov_logdets)
    
    # compute inverse cov, can this be done in a more clever way?
    # ideas for computing Z more efficiently maybe
    # einsum first line of eq. 19 over "big" indices
    # weighted scalar product in numpy?
    
#    # zinv by pieces
#    inv_small_ns = la.inv(small_ns_mat)
#    Zinv = np.zeros((P*N, P*N))
#    for A, B in np.ndindex(Zinv.shape):
#        k = A//N
#        alpha = A%N
#        q = B//N
#        beta = B%N
#        
#        element = 0
#        for s, inv_sig in enumerate(self._TOA_FD_inv_covs):
#            element += inv_small_ns[s, k] * inv_sig[alpha, beta] * inv_small_ns[s, q]
#    
#        Zinv[A, B] = element
    
    # direct inverse (this is faster)
    Zinv = la.inv(Z)
    
    return big_ns_mat, Z, Zinv, log_det_Z


def log_likelihood_FD_ns(self, source, model_func, model_args, **model_kwargs):
    """
    FD log likelihood using null-streams
    """
    P = self._n_pulsars
    N = self._n_freqs
    
    # get the null_stream matrix
    pulsar_array = self._pulsars[['theta', 'phi']].values
    ns_mat = construct_M(*source, pulsar_array) # the "small" ns matrix
    
    # get big (concatenated) ns matrix. ns covariance, its invserve and its determinant
    big_ns_mat, ns_cov, inv_ns_cov, log_det_ns_cov = self._ns_covariance(ns_mat)
    
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
    ll = -0.5 * (np.einsum('i,ij,j', x, inv_ns_cov, np.conj(x)))
    # no 0.5 in norm because complex quantity
    norm = N*P*np.log(2*np.pi) - log_det_ns_cov
    
    assert(abs(np.imag(ll)) < abs(np.real(ll) * 1e-10))
    ll = np.real(ll)
    
    return ll #+ norm


functions = [concatenate_residuals, _ns_covariance, log_likelihood_FD_ns]