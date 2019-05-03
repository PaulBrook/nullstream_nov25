#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:24:19 2019

@author: jgoldstein
"""

import numpy as np
import numpy.random as rd
import pandas as pd

from jannasutils import radec_location_to_ang

class PTA_sim:
    def __init__(self):
        self.pulsars = pd.DataFrame(columns=['theta', 'phi', 'rms', 'nTOA'])
        self.times = None
        self.residuals = None
        
        
    def random_pulsars(self, n, mean_rms=1e-6, sig_rms=0, overwrite=False):
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
        try:
            assert pulsar_locations.shape[1] == 2
            assert pulsar_locations.shape[0] == rms.shape[0]
        except: 
            raise ValueError('pulsar_locations or rms array not the right shape')
        
        self.pulsars[['theta', 'phi']] = pulsar_locations
        self.pulsars['rms'] = rms
        
        
    def pulsars_from_file(self, filepath='/home/jgoldstein/Documents/PTA/nullstream/code/clean_code/PTA_files/IPTA_pulsars.txt'):
        PTAdata = np.loadtxt(filepath, skiprows=1, comments='#')
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
        
    
        
        
        
        
        
    
    
    
        