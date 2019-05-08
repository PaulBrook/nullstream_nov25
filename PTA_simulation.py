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
import matplotlib.pyplot as plt

from jannasutils import radec_location_to_ang
from nullstream_algebra import response_matrix

# all times are in seconds (or 1/seconds)
YEAR = 365.25 * 24 * 3600

class PTA_sim:
    def __init__(self):
        self._pulsars = pd.DataFrame(columns=['theta', 'phi', 'rms', 'nTOA'])
        self._n_pulsars = 0
        self._times = 0
        self._hplus = 0
        self._hcross = 0
        self._signal= 0
        self._noise = 0
        
    @property
    def residuals(self):
        return self._signal + self._noise
        
    def _check_empty_pulsars(self, overwrite=False):
        if not self._pulsars.empty:
            if overwrite:
                # empty previous data
                self._pulsars.drop(index=self._pulsars.index, inplace=True)
            else:
                raise ValueError('Already have pulsar data, specify overwrite=True')
        
        
    def random_pulsars(self, n, mean_rms=1e-7, sig_rms=0, overwrite=False):
        """
        Pick n random pulsars, uniformly distributed over the sky.
        
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
        overwrite: If true, overwrite already existing pulsars with new ones
            default = False
        """
        self._check_empty_pulsars(overwrite=overwrite)
        self._n_pulsars = n
                
        # random locations on the sphere
        random_ab = rd.rand(n, 2)
        self._pulsars['theta'] = np.arccos(random_ab[:, 0]*2 - 1)
        self._pulsars['phi'] = random_ab[:, 1] * 2 * np.pi
        
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
        
        
    def plot_pulsar_map(self):
        zero_map = np.zeros(hp.nside2npix(1))
        hp.mollview(zero_map, title='{} pulsar PTA'.format(len(self._pulsars)))
        
        marker_sizes = (self._pulsars['rms'].values/1.e-7)**(-0.4)*10
        for p, pulsar in enumerate(self._pulsars[['theta', 'phi']].values):
            hp.projplot(*pulsar, marker='*', c='w', ms=marker_sizes[p])
            
    
    def evenly_sampled_times(self, cadence=1e6, T=20*YEAR, t_start=0):
        """
        Set the same evenly sampled times for all pulsars.
        """
        times = np.arange(t_start, T, cadence)
        self._pulsars['nTOA'] = len(times)
        self._times = np.array((times,)*self._n_pulsars)
        
        # create zero residuals for now (inject signal/make noise later)
        self._hplus = np.zeros_like(times) # single row (same for all pulsars)
        self._hcross = np.zeros_like(times)
        
        self._signal = np.zeros_like(self._times) # row per pulsar
        self._noise = np.zeros_like(self._times)
        
    
    def randomized_times(self, mean_cadence=1e6, meanT=20*YEAR, t_start=0):
        """
        Set somewhat randomized observation times with average cadence, within#
        the same observation time for all pulsars.
        """
        # TODO
        pass
    
    
    def gappy_times(self):
        """
        Set times with random gaps and within different observation windows 
        for all pulsars.
        """
        # TODO
        pass
    
    def times_from_tim_file(self, filepath):
        """
        Read pulsar times from .tim file (such as from IPTA data release)
        """
        # TODO
        pass
    
    def inject_signal(self, signal_func, source, *signal_args, **signal_kwargs):
        """
        Inject signal into the residuals.
        """
        self._hplus, self._hcross = signal_func(self._times, *signal_args, **signal_kwargs)
        responses = response_matrix(*source, self._pulsars[['theta', 'phi']].values)
        Fplus = np.expand_dims(responses[:, 0], -1)
        Fcross = np.expand_dims(responses[:, 1], -1)
        # add to signal in case we want multiple injections done
        self._signal += Fplus * self._hplus + Fcross * self._hcross
        
    def white_noise(self):
        """
        Inject gaussian noise according to each pulsar's rms level. 
        This deletes any previously injected noise (but keeps signal the same).
        """
        # annoyingly, rd.normal cannot handle both an array for the scale values
        # and more than a scalar output for each of those values, so we have to
        # loop through the pulsars
        nTOAs = self._times.shape[-1]
        noise = [rd.normal(scale=self._pulsars['rms'][i], size=nTOAs) 
                 for i in range(sim._n_pulsars)]
        
        self._noise = np.array(noise)
        assert(self._noise.shape == self._times.shape)
        # don't add to any previously existing noise
        self._noise = noise
        
        
    def plot_residuals(self):
        """
        Plot times vs residuals for all pulsars
        """
        fig, ax = plt.subplots(1)
        ax.plot(self._times.T, self.residuals.T, ls='none', marker='.')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('residuals (s)')
        
    def clear_residuals(self):
        """
        Delete any signal and noise from the residuals.
        """
        self._hplus = None
        self._hcross = None
        self._signal = np.zeros_like(self._times)
        self._noise = np.zeros_like(self._times)
        
        
if __name__ == '__main__':
    print('An example of PTA sim')
    # make a simulation object (we may want to have initialisation options that
    # automatically do the next few steps, but for now we do them by hand)
    sim = PTA_sim()
    
    # make some pulsars, in this case 5 random ones with some variation in rms
    # and plot a skymap (bigger markers are better pulsars)
    sim.random_pulsars(5, sig_rms=5e-8)
    sim.plot_pulsar_map()
    
    # set some evenly sampled times (default options)
    sim.evenly_sampled_times()
    
    # inject a sinusoid signal
    # arguments are: phase, amplitude, polarization, cos(i), GW frequency (rd/s)
    from GW_models import sinusoid_TD
    GW_args = [0.1, 1e-14, np.pi/7, 0.3, 1e-8] 
    source = (0.8 * np.pi, 1.3 * np.pi)
    sim.inject_signal(sinusoid_TD, source, *GW_args)
    
    # inject white noise
    sim.white_noise()
    
    # not plot the residuals
    sim.plot_residuals()
    