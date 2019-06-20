#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:39:00 2019

@author: jgoldstein

functions to go in the main PTA_sim class that have to do with picking/setting the pulsars
"""

import numpy as np
import numpy.random as rd
import healpy as hp

try:
    from jannasutils import radec_location_to_ang
except:
    # use hacked excerpt from jannasutils
    from from_jannasutils import radec_location_to_ang


def _check_empty_pulsars(self, overwrite=False):
    if not self._pulsars.empty:
        if overwrite:
            # empty previous data
            self._pulsars.drop(index=self._pulsars.index, inplace=True)
        else:
            raise ValueError('Already have pulsar data, specify overwrite=True')

def random_pulsars(self, n, mean_rms=1e-7, sig_rms=0, uniform=True,
                   overwrite=False):
    """
    Pick n random pulsars from across the sky.

    Parameters
    ----------
    n: int
        number of pulsars to pick
    mean_rms: float
        mean rms of the residuals of each pulsar
        default = 1e-7 s (100 ns)
    sig_rms: if not zero, pick rms values from a gaussian distribution
        with this as its standard deviation (except negative values
        are mapped to their positive counterpart)
        default = 0
    uniform: If true, draw pulsars evenly across the sky. Otherwise, choose
        from a distribution weighted by the population of known msps
    overwrite: If true, overwrite already existing pulsars with new ones
        default = False
    """
    self._check_empty_pulsars(overwrite=overwrite)
    self._n_pulsars = n

    if uniform:
        # random locations on the sphere
        random_ab = rd.rand(n, 2)
        self._pulsars['theta'] = np.arccos(random_ab[:, 0]*2 - 1)
        self._pulsars['phi'] = random_ab[:, 1] * 2 * np.pi
    else:
        # draw randomly from weighted set of healpix pixels (see Roebber 2019)
        # FIXME find this file with relative path from the package or something?
        weights = np.loadtxt('msp_weight_map.dat')
        npix = np.size(weights)
        nside = hp.npix2nside(npix)
        pix = np.random.choice(npix, n, replace=False, p=weights)
        self._pulsars['theta'], self._pulsars['phi'] = hp.pix2ang(nside, pix)

    # normal distribution of rms values
    self._pulsars['rms'] = abs(rd.normal(loc=mean_rms, scale=sig_rms, size=n))

    # save the inverse covariance matrix of the pulsar residuals (Time Domain)
    self._inv_cov_residuals = np.diag(1/self._pulsars['rms'])

def set_pulsars(self, pulsar_locations, rms, overwrite=False):
    """
    Set pulsars with array of locations and specified residual rms values.

    Parameters
    ----------
    pulsar_locations: numpy array
        2xn array of (theta, phi) coordinates for n pulsars
        theta is the polar coordinate between 0 and pi,
        phi is the azimuthal coordinate between 0 and 2 pi
    rms: numpy array
        length n array with an rms value (in seconds) for each pulsar
    overwrite: If true, overwrite already existing pulsars with new ones
        default = False
    """
    self._check_empty_pulsars(overwrite=overwrite)

    try:
        assert pulsar_locations.shape[1] == 2
        n = pulsar_locations.shape[0]
        assert rms.shape[0] == n
    except:
        raise ValueError('pulsar_locations or rms array not the right shape')

    self._n_pulsars = n
    self._pulsars['theta'] = pulsar_locations[:, 0]
    self._pulsars['phi'] = pulsar_locations[:, 1]
    self._pulsars['rms'] = rms

    # save the inverse covariance matrix of the pulsar residuals (Time Domain)
    self._inv_cov_residuals = np.diag(1/self._pulsars['rms'])

def pulsars_from_file(self, filepath='./PTA_files/IPTA_pulsars.txt',
                      skip_lines=1, overwrite=False):
    """
    Load pulsar locations and rms values from text file.

    Parameters
    ----------
    filepath: path the the PTA file
        The PTA file must have in it's first five columns:
        Ra (hours), Ra (minutes), Dec (degrees), Dec (arcminutes), rms (seconds)
        default = './PTA_files/IPTA_pulsars.txt'
    skip_lines: int
        Number of lines to skip when reading in the PTA file (header lines)
        default = 1
    overwrite: If true, overwrite already existing pulsars with new ones
        default = False
    """
    self._check_empty_pulsars(overwrite=overwrite)

    PTAdata = np.loadtxt(filepath, skiprows=skip_lines, comments='#')
    self._n_pulsars = len(PTAdata)

    # get pulsar ra hour, ra min, dec degree, dec arcs from column 0123
    # and convert to theta, phi
    PTApulsars = PTAdata[:, 0:4]
    # convert Ra Dec in PTA data to theta, phi
    pulsars_ang = np.array([radec_location_to_ang(PTApulsars[i]) for i in range(self._n_pulsars)])
    self._pulsars['theta'] = pulsars_ang[:, 0] # for some reason can't do both these lines in one
    self._pulsars['phi'] = pulsars_ang[:, 1]

    # get rms from column 4 and convert microseconds in PTA data to seconds
    self._pulsars['rms'] = 1.0e-6 * PTAdata[:, 4]

    # save the inverse covariance matrix of the pulsar residuals
    self._inv_cov_residuals = np.diag(1/self._pulsars['rms'])


def plot_pulsar_map(self):
    zero_map = np.zeros(hp.nside2npix(1))
    hp.mollview(zero_map, title='{} pulsar PTA'.format(len(self._pulsars)))

    marker_sizes = (self._pulsars['rms'].values/1.e-7)**(-0.4)*10
    for p, pulsar in enumerate(self._pulsars[['theta', 'phi']].values):
        hp.projplot(*pulsar, marker='*', c='w', ms=marker_sizes[p])


# functions we want to add as methods to the main PTA_sim class
functions = [_check_empty_pulsars, random_pulsars, set_pulsars, pulsars_from_file, plot_pulsar_map]