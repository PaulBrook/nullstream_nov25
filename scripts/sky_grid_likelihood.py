#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:10:27 2020

@author: jgoldstein
"""
import argparse
import yaml
import pickle

import numpy as np
import healpy as hp
import pandas as pd

import matplotlib.pyplot as plt
from os.path import join

###### command line options ###
#parser = argparse.ArgumentParser("Compute NS null likelihood over a grid on the sky")
#parser.add_argument('-p', '--sim_pickle', required=True, dest='sim_pickle',
#                    help='pta_sim.pickle file to load from')
#parser.add_argument('-o', '--output_dir', default='./grid_output/', dest='out_dir',
#                    help='directory to save plot in')
#args = parser.parse_args()

sim_pickle='/home/jgoldstein/Documents/projects/ptacake_runs/sky_runs/NSnull_nonoise/10pulsars/output/pta_sim.pickle'
nside=24
fig_path = '/home/jgoldstein/Documents/projects/ptacake_runs/sky_runs/NSnull_nonoise/10pulsars/grid_output'

def Nsnull_loglike(sim, source):
    loglike = sim.log_likelihood_FD_onlynull(source, add_norm=False, return_only_norm=False)
    return loglike

if __name__ == '__main__':
    try:
        with open(sim_pickle, 'rb') as f1:
            sim = pickle.load(f1)
    except FileNotFoundError:
        raise ValueError('Could not find pta sim pickle file at {}'.format(args.sim_pickle))
        
    
    # create healpy map of pixels to compute loglike over
    npix = hp.nside2npix(nside)
    pixels = pd.DataFrame(index=np.arange(npix))
    
    # compute theta, phi coordinates from pix numbers
    theta, phi = hp.pix2ang(nside, pixels.index.values)
    pixels['theta'] = theta
    pixels['phi'] = phi

    # compute log likelihood at each pix
    pixels['loglike'] = np.zeros(npix)
    for ipix in pixels.index:
        source = [pixels['theta'].iloc[ipix], pixels['phi'].iloc[ipix]]
        pixels['loglike'].iloc[ipix] = Nsnull_loglike(sim, source)
        
    # make a healpy loglike and a like map
    hp.mollview(pixels['loglike'].values, title='Ns null log likelihood', cbar=True)
    fig = plt.gcf()
    fig.savefig(join(fig_path, 'loglike_map.pdf'))
    
    hp.mollview(10**pixels['loglike'].values, title='Ns null likelihood', cbar=True)
    fig2 = plt.gcf()
    fig2.savefig(join(fig_path, 'like_map.pdf'))
        
    