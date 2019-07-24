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
            
    
    def log_likelihood_FD_ns(self, source, model_func, model_args, **model_kwargs):
        """
        FD log likelihood using null-streams
        """
        P = self._n_pulsars
        N = self._n_freqs
        
        # get the null_stream matrix
        pulsar_array = self._pulsars[['theta', 'phi']].values
        ns_mat = construct_M(*source, pulsar_array) # the "small" ns matrix
        big_ns_mat = np.zeros((P*N, P*N))
        for j in range(N):
            # starting from point [j, j], put the elements of the small ns_mat
            # at every point [j + a*N, j + b*N], with a and b any integers.
            big_ns_mat[j::N, j::N] = ns_mat
            
        # make null-streams out of concatenated residuals
        null_streams = np.einsum('ij,j', big_ns_mat, self.residuals_concat)
        
        
        # compute ns covariance etc
        inv_ns_mat = la.inv(ns_mat)
        
        # how can we compute the (inverse) covariance of the concatenated null-streams?
        # use Z = covariance of the concatenated, FD null-streams
        # I don't know if either of the following methods are correct (but definitely not both)
        # for-loop method?
        Zinv = np.zeros((P*N, P*N))
        for A, B in np.ndindex(Zinv.shape):
            k = A//N
            alpha = k%N
            l = B//N
            beta = B%N
            
            element = 0
            for s, inv_sig in self._TOA_FD_inv_covs:
                element += np.conj(inv_ns_mat[s, l]) * inv_sig[beta, alpha] * inv_ns_mat[s, k]
            Zinv[A, B] = element
            
        # einsum method?
        FD_inv_covs = np.array(self._TOA_FD_inv_covs)
        Zinv_einsum = np.einsum('sl,sba,sk', np.conj(inv_ns_mat), FD_inv_covs, inv_ns_mat)
        
        
        # compute determinant of the ns covariance matrix
        sign, log_det_ns_mat = la.slogdet(ns_mat)
        log_det_Z = 2*N*log_det_ns_mat + np.sum(self._TOA_FD_cov_logdets)


        # call model function with args and preset model times, then funky fourier
        fourier_hplus, fourier_hcross = self.fourier_model(model_func, *model_args, **model_kwargs)
        # combine hplus, hcross with null streams to get full model (concatenated)
        nulls = (np.zeros_like(fourier_hplus),)*(self._n_pulsars-2)
        ns_model = np.concatenate([fourier_hplus, fourier_hcross, *nulls])
        
        # compute product of data - model
        x = null_streams - ns_model
        ll = -0.5 * np.einsum('i,ij,j', x, Zinv, np.conj(x))
        # no 0.5 in norm because complex quantity
        norm = N*P*np.log(2*np.pi) - log_det_Z
        return ll #+ norm


functions = []