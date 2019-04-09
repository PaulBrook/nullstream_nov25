# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:23:52 2017

@author: jgoldstein

PTA response functions
and construction of M matrix from null-stream paper
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
    

def responses(theta_s, phi_s, pulsars):
    """
    Get response functions for a pulsar and a GW source location
    
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
    return Fplus, Fcross

### CHECKED/IMPROVED UP TO HERE ####
    
def response_matrix(theta_s,phi_s, pulsars):
    """
    Column matrix of response functions
    
    N x 2 matrix with Fplus, Fcross response functions in each row, 
    for N pulsars. Uses responses() to calculate Fplus, Fcross
    
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
        N x 2 matrix of response functions
        N is the number of pulsars in the PTA (the length of pulsars)
    """
    R = [ list(responses(theta_s, phi_s, p[0], p[1])) for p in pulsars]
    return np.array(R)
    
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
    N = len(pulsars)    
    
    R = response_matrix(theta_s, phi_s, pulsars)
    RinvMP = la.pinv(R)
    # construct projection matrix for the null space
    null_matrix = np.identity(N) - np.einsum('ij,jk',R,RinvMP)
    
    # use QR-decomposition to find orthonormal basis of null space
    # first N-2 columns form basis
    q,r = la.qr(null_matrix)
    
    # construct "super" F matrix
    M = np.zeros((N,N),dtype='complex')
    # the second to the last row project onto null space using orthonormal basis
    # of the null space as constructed above
    for i in range(N-2):    
        M[2+i,:] = q[:,i]
    # first two rows transform redshifts back to strain
    M[0,:] = RinvMP[0,:]
    M[1,:] = RinvMP[1,:]
    
    return np.real(M)
    
def Fisher_matrix(A, S):
    """
    Calculate Fisher matrix from transformation matrix A and covariance matrix S
    
    For a quantity y = A x, and covariance matrix S for x, the covariance matrix
    for y is given by Sy = A S A^T. (A^T is the transpose of A). Then the Fisher
    matrix is given by (Sy)^-1 = (A^-1)^T (S^-1) (A^-1). 
    
    Parameters
    ----------
    A: NumPy Array
        NxN transformation matrix (y = A x)
    S: NumPy Array
        NxN covariance matrix (for x)
        
    Returns
    -------
    Numpy Array
        NxN Fisher matrix (for y)
        
    Example
    -------
    Given covariance matrix S for redshifts z, and a transformation to a 
    strain + null stream vector h = M z, the Fisher matrix for h is given by
    (M^-1)^T (S^1) (M^-1)
        
    """
    Sinv = la.inv(S)
    Ainv = la.inv(A)
    # transpose is done by swapping indices in the einsum on the first matrix
    return np.einsum('ji,jk,kl', Ainv, Sinv, Ainv)