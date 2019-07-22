"""
Created on Tue Jun 18 14:50:47 2019

@author: jgoldstein
"""
import numpy as np

from .nullstream_algebra import null_streams, response_matrix

def _setup_nullstreams(self):
    
    # check that fourier residuals etc have been set up
    assert self._freqs is not None, 'Null-streams only in Fourier Domain, run fourier_residuals first'
    
    # re-order residuals in chuncks per frequency, all in one vector
    self._signal_by_freq = self._signalFD.T.flatten()
    self._noise_by_freq = self._noiseFD.T.flatten()
    
    # reorder covariance matrices so that we have one PxP matrix per frequency,
    # rather than one NxN matrix per pulsar
    # assume all covariance matrices are diagonal so we just need to consider the diagonal elements
    # TODO  are they also diagonal when the noise rms changes over time?
    FD_covs_diagonals = [np.diag(cov) for cov in self._TOA_FD_covs]
    diagonals_by_frequency = np.array(FD_covs_diagonals).T
    self._FD_cov_by_frequency = [np.diag(d) for d in diagonals_by_frequency]
    
    
    


functions = []