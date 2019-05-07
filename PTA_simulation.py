#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:24:19 2019

@author: jgoldstein
"""

import numpy as np
import numpy.random as rd
import pandas as pd
import healpy as hp

from jannasutils import radec_location_to_ang

class PTA_sim:
    def __init__(self):
        self.pulsars = pd.DataFrame(columns=['theta', 'phi', 'rms', 'nTOA'])
        self.times = None
        self.residuals = None
        
        
    def random_pulsars(self, n, mean_rms=1e-6, sig_rms=0, overwrite=False):
        """
        Pick n random pulsars, uniformly distributed over the sky.
        
        Parameters
        ----------
        n: int
            number of pulsars to pick
        mean_rms: float
            mean rms of the residuals of each pulsar
            default = 1e-6 s
        sig_rms: if not zero, pick rms values from a gaussian distribution 
            with this as its standard deviation
            default = 0
        overwrite: If true, overwrite already existing pulsars with new ones
            default = False
        """
        if not self.pulsars.empty:
            if overwrite:
                # empty previous data
                self.pulsars.drop(index=self.pulsars.index, inplace=True)
            else:
                raise ValueError('Already have pulsar data, specify overwrite=True')
                
        # random locations on the sphere
        random_ab = rd.rand(n, 2)
        self.pulsars['theta'] = np.arccos(random_ab[:, 0]*2 - 1)
        self.pulsars['phi'] = random_ab[:, 1] * 2 * np.pi
        
        # normal distribution of rms values
        self.pulsars['rms'] = rd.normal(loc=mean_rms, scale=sig_rms, size=n)
        
        
    def set_pulsars(self, pulsar_locations, rms):
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
        """
        try:
            assert pulsar_locations.shape[1] == 2
            assert pulsar_locations.shape[0] == rms.shape[0]
        except: 
            raise ValueError('pulsar_locations or rms array not the right shape')
        
        self.pulsars[['theta', 'phi']] = pulsar_locations
        self.pulsars['rms'] = rms
        
        
    def pulsars_from_file(self, filepath='./PTA_files/IPTA_pulsars.txt', skip_lines=1):
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
        """
        PTAdata = np.loadtxt(filepath, skiprows=skip_lines, comments='#')
        N = len(PTAdata)
        
        # get pulsar ra hour, ra min, dec degree, dec arcs from column 0123
        # and convert to theta, phi
        PTApulsars = PTAdata[:, 0:4]
        # convert Ra Dec in PTA data to theta, phi
        pulsars_ang = np.array([radec_location_to_ang(PTApulsars[i]) for i in range(N)])
        self.pulsars['theta'] = pulsars_ang[:, 0] # for some reason can't do both these lines in one
        self.pulsars['phi'] = pulsars_ang[:, 1]
        
        # get rms from column 4 and convert microseconds in PTA data to seconds
        self.pulsars['rms'] = 1.0e-6 * PTAdata[:, 4]
        
        
    def plot_pulsar_map(self):
        zero_map = np.zeros(hp.nside2npix(1))
        hp.mollview(zero_map, title='{} pulsar PTA'.format(len(self.pulsars)))
        
        marker_sizes = (self.pulsars['rms'].values/1.e-7)**(-0.4)*10
        for p, pulsar in enumerate(self.pulsars[['theta', 'phi']].values):
            hp.projplot(*pulsar, marker='*', c='w', ms=marker_sizes[p])
    
    
    
        