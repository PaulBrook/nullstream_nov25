# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:23:52 2017

@author: jgoldstein

- PTA response functions
- construction of M matrix from null-stream paper
 and inverse covariance matrix of transformed quantity
- apply null-stream transform to single-frequency data

All checked/imporved 30/04/19
"""

import numpy as np
import numpy.linalg as la

def position_vector(theta, phi):
    """
    Get unit vector pointing to sky location theta, phi
    
    Parameters
    ----------
    theta: float
        polar angle of the position in radians, between 0 and pi
    phi: float
        azimuthal angle of the position in radians, between 0 and 2 pi
        
    Returns
    -------
    NumPy Array
        Unit vector pointing to the input sky locations
    """
    Omega = [np.sin(theta) * np.cos(phi), 
             np.sin(theta) * np.sin(phi), np.cos(theta)]
    return np.array(Omega)
    
    
def source_polarisation_tensors(theta, phi):
    """
    Get polarisation tensors associated with coordinates theta, phi
    (theta, phi) is the point the GW is travelling towards
    
    Definitions taken from the Anholm et al. paper
    http://arxiv.org/abs/0809.0701
    
    Parameters
    ----------
    theta: float
        polar angle of the position in radians, between 0 and pi
    phi: float
        azimuthal angle of the position in radians, between 0 and 2 pi
        
    Returns:
    NumPy Array
        plus polarisation tensor
    NumPy Array
        cross polarisation tensor
    """          
    m = [np.sin(phi), -np.cos(phi), 0]
    n = [np.cos(theta) * np.cos(phi), \
    np.cos(theta) * np.sin(phi), -np.sin(theta)]
    
    # find cool matrix way of constructing these
    eplus = np.array([[m[i]*m[j] - n[i]*n[j] for j in range(3)] for i in range(3)])
    ecross = np.array([[m[i]*n[j] + n[i]*m[j] for j in range(3)] for i in range(3)])
    return eplus, ecross
    

def response_matrix(theta_s, phi_s, pulsars):
    """
    Get matrix of response functions given the source location and all pulsars.
    
    First column is Fplus per pulsar, second column is Fcross per pulsar. 
    Number of rows equal to the number of pulsars. 
    Plus and cross response functions accroding to the definitions
    in the Anholm et al. paper http://arxiv.org/abs/0809.0701
    
    Parameters
    ----------
    theta_s: float
        polar angle of the source position in radians, between 0 and pi
    phi_s: float
        azimuthal angle of the source position in radians, between 0 and 2 pi
    pulsars: numpy array
        N x 2 array of theta, phi coordinates of pulsars
        theta: polar angle of the pulsar position in radians, between 0 and pi
        phi: azimuthal angle of the pulsar position in radians, between 0 and 2 pi
        
    Returns
    -------
    numpy array
        Plus response function value for each pulsar
    numpy array
        Cross response function value for each pulsar
    """
    source = position_vector(theta_s, phi_s)
    # We need the vector pointing in the GW propagation direction,
    # so from the source not toward it
    omega = - source
    p = position_vector(pulsars[:, 0], pulsars[:, 1])  
    # use antipodal point to source to construct polarisation tensors
    eplus, ecross = source_polarisation_tensors(np.pi - theta_s, np.pi + phi_s)
    
    prefactor = 0.5 / (1 + np.dot(omega, p))
    # using einstein summation convention, the product for an individual pulsar
    # is p_i eplus_ij p_j, the '...' is the extra dimension for the number of pulsars
    Fplus = prefactor * np.einsum('i...,ij,j...', p, eplus, p)
    Fcross = prefactor * np.einsum('i...,ij,j...', p, ecross, p)
    
    return np.vstack((Fplus, Fcross)).T

    
def construct_M(theta_s, phi_s, pulsars):
    """
    Get the full M matrix the "super" matrix)
    
    Construct the combination M of the response matrix and orthogonal null 
    stream projection matrix. M projects redshifts onto a vector of strain 
    and nullstreams.
    Say R is the column matrix of response functions (from response_matrix())
    that transform strain to redshifts. The first two rows of M are the 
    pseudo-inverse of R. Taking the inverse of M gives a matrix which has R as
    its first two columns. (This inverse will project a strain + null stream
    vector onto redshifts.)
    
    Parameters
    ----------
    theta_s: float
        polar angle of the source position in radians, between 0 and pi
    phi_s: float
        azimuthal angle of the source position in radians, between 0 and 2 pi
    pulsars: list
        List of coordinates of the pulsars in the PTA, where each coordinate
        is an array with theta, phi (polar and azimuthal coordinate)
    
    Returns
    -------
    NumPy Array
        N x N super matrix that projects N redshifts onto 2 strains and
        N-2 null streams
    """
    n = len(pulsars)

    R = response_matrix(theta_s, phi_s, pulsars)
    # get Moore-Penrose inverse (= pseudo inverse) of R
    Rinv = la.pinv(R)
    
    # Get null space projection matrix, then use qr decomposition to 
    # get n-2 basis vectors of null space (first n-2 columns of q)
    null_projection = np.eye(n) - np.einsum('ij,jk', R, Rinv)
    q,r = la.qr(null_projection)
    null_basis = q[:, np.arange(n) < (n-2)]
    
    # construct "super" M matrix. First two rows: MP inverse of R
    # last n-2 rows: transposed null basis
    M = np.vstack((Rinv, null_basis.T))

    return M

def inv_transformed_cov(T, Cinv):
    """
    Calculate inverse covariance matrix for a transformed quantity.
    
    For a transformed quantity y = T x, where x has covariance matrix C, y has
    covariance matrix T C Tt (Tt being the transpose of T). Then the inverse
    covariance matrix of y is given by (T C Tt)^(-1) = T^(-t) C^(-1) T^(-1).
    
    Parameters
    ----------
    T: numpy array
        Transformation matrix (null-stream super matrix) NxN
    Cinv: numpy array
        Inverse covariance matrix of the un-transformed quantity NxN
        
    Returns
    -------
    numpy array
        Inverse covariance matrix of the transformed quantity NxN
    """
    Tinv = la.inv(T)
    # transpose of the first term in the product is done by swapping the order
    # of its indices in the einstein summation convention
    return np.einsum('ji,jk,kl', Tinv, Cinv, Tinv)

def null_streams(data, invC, source, pulsars):
    """
    Construct signal streams + null streams from TD or FD data
    
    Data and pulsars are numpy arrays with the same length N (first dimension), 
    i.e. we have a data stream for each pulsar. The data stream can be in time
    or frequency domain, or can be a single number for single frequency data.
    
    Parameters
    ----------
    data: numpy array
        (N x ndata) data stream (or point) for each pulsar
    invC: numpy array
        (N x N) inverse covariance matrix for the data
    source: numpy array
        source location (theta, phi)
    pulsars: numpy array
        N x 2 array of theta, phi coordinates of the N pulsars
        theta: polar angle of the pulsar position in radians, between 0 and pi
        phi: azimuthal angle of the pulsar position in radians, between 0 and 2 pi
    
    Returns
    -------
    numpy array
        first two entries are reconstructed signal, then N-2 null streams 
        (where N is the number of pulsars)
    numpy array
        (N x N) transformed inverse covariance matrix to go with null stream data
    """
    M = construct_M(*source, pulsars)
    null_streams = np.dot(M, data)
    inv_cov = inv_transformed_cov(M, invC)
    return null_streams, inv_cov
    