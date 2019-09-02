#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:39:00 2019

@author: jgoldstein

functions to go in the main PTA_sim class that have to do with picking/setting the pulsars
"""
import os
import numpy as np
import numpy.random as rd
import healpy as hp
import matplotlib.pyplot as plt

# for reading in package data
try:
    # Python 3.7+
    import importlib.resources as import_resources
except ImportError:
    # Backported version in Python < 3.7
    import importlib_resources as import_resources

try:
    from jannasutils import radec_location_to_ang
except:
    # use hacked excerpt from jannasutils
    from .from_jannasutils import radec_location_to_ang


def _check_empty_pulsars(self, overwrite=False):
    if not self._pulsars.empty:
        if overwrite:
            # empty previous data
            self._pulsars.drop(index=self._pulsars.index, inplace=True)
        else:
            raise ValueError('Already have pulsar data, specify overwrite=True')

def random_pulsars(self, n, mean_rms=1e-7, sig_rms=0, uniform=True,
                   weight_map_dir=None, overwrite=False, seed=None):
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
    weight_map_dir: None or str (path)
        If not None, look for 'msp_weight_map.dat' (used for non-uniform pulsars)
        in this direcotory. Otherwise try to find it in 'ptacake'
    overwrite: If true, overwrite already existing pulsars with new ones
        default = False
    seed: None or int
        default = None
        if not None, use np.random.seed(seed) to get repeatable random behaviour
    """
    self._check_empty_pulsars(overwrite=overwrite)
    self._n_pulsars = n
    
    if seed is not None:
        rd.seed(seed)

    if uniform:
        # random locations on the sphere
        random_ab = rd.rand(n, 2)
        self._pulsars['theta'] = np.arccos(random_ab[:, 0]*2 - 1)
        self._pulsars['phi'] = random_ab[:, 1] * 2 * np.pi
    else:
        # draw randomly from weighted set of healpix pixels (see Roebber 2019)
        if weight_map_dir:
            f = os.path.join(weight_map_dir, 'msp_weight_map.dat')
            weights = np.loadtxt(f)
        else:
            with import_resources.path('ptacake', 'msp_weight_map.dat') as f:
                weights = np.loadtxt(f)
        npix = np.size(weights)
        nside = hp.npix2nside(npix)
        pix = np.random.choice(npix, n, replace=False, p=weights)
        self._pulsars['theta'], self._pulsars['phi'] = hp.pix2ang(nside, pix)

    # normal distribution of rms values
    self._pulsars['rms'] = abs(rd.normal(loc=mean_rms, scale=sig_rms, size=n))


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


def plot_pulsar_map(self, plot_point=None):
    """
    Plot map of the pulsars. Bigger pulsars have lower residuals rms.
    Optional plot_point = (theta, phi) plots this points as a cross on the map.
    """
    zero_map = np.zeros(hp.nside2npix(1))
    hp.mollview(zero_map, title='{} pulsar PTA'.format(len(self._pulsars)))

    marker_sizes = (self._pulsars['rms'].values/1.e-7)**(-0.4)*10
    for p, pulsar in enumerate(self._pulsars[['theta', 'phi']].values):
        hp.projplot(*pulsar, marker='*', c='w', ms=marker_sizes[p])
    if plot_point is not None    :
        hp.projplot(*plot_point, marker='+', c='w', ms=10)
    
    return plt.gcf()


# functions we want to add as methods to the main PTA_sim class
functions = [_check_empty_pulsars, random_pulsars, set_pulsars, pulsars_from_file, plot_pulsar_map]
