#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:10:27 2020

@author: jgoldstein

plot likelihood maps of null-stream runs

for a given sky-run with N pulsars, using likelihood Q (null-stream null only),
- load the pickled PTA_sim, which can be used to compute the (log)likelihood
- make a healpy map of the likelihood and log likelihood
- make the same maps with overplotted posterior samples

"""
import argparse
import pickle
import os

import numpy as np
import healpy as hp
import pandas as pd

import matplotlib.pyplot as plt
from os.path import join

from pp_utils import get_post

####### command line options ###
#parser = argparse.ArgumentParser("Compute NS null likelihood over a grid on the sky")
#parser.add_argument('-p', '--sim_pickle', required=True, dest='sim_pickle',
#                    help='pta_sim.pickle file to load from')
#parser.add_argument('-P', '--posterior', required=True, dest='post_file
#                    help='posterior.dat file from cpnest')
#parser.add_argument('-o', '--output_dir', default='./grid_output/', dest='out_dir',
#                    help='directory to save plots in')
#args = parser.parse_args()
#
#sim_pickle = args.sim_pickle
#fig_path = args.out_dir
#post_file = args.post_file

sim_pickle='/home/jgoldstein/Documents/projects/ptacake_runs/sky_runs/NSnull_nonoise/8pulsars/output/pta_sim.pickle'
fig_path = '/home/jgoldstein/Documents/projects/ptacake_runs/sky_runs/NSnull_nonoise/8pulsars/grid_output'
post_file = '/home/jgoldstein/Documents/projects/ptacake_runs/sky_runs/NSnull_nonoise/8pulsars/output/posterior.dat'

nside=24

if not os.path.exists(fig_path):
    os.mkdir(fig_path)

def Nsnull_loglike(sim, source):
    loglike = sim.log_likelihood_FD_onlynull(source, add_norm=False, return_only_norm=False)
    return loglike

if __name__ == '__main__':
    try:
        with open(sim_pickle, 'rb') as f1:
            sim = pickle.load(f1)
    except FileNotFoundError:
        raise ValueError('Could not find pta sim pickle file at {}'.format(sim_pickle))
        
    
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
    hp.mollview(pixels['loglike'].values, title='Ns null log likelihood', 
                cbar=True, cmap='inferno')
    fig = plt.gcf()
    fig.savefig(join(fig_path, 'loglike_map.pdf'))
    
    hp.mollview(10**pixels['loglike'].values, title='Ns null likelihood', 
                cbar=True, cmap='inferno')
    fig2 = plt.gcf()
    fig2.savefig(join(fig_path, 'like_map.pdf'))
    
    # read in posterior 
    post_samples = get_post(post_file)
    post_samples['theta'] = np.arccos(post_samples['costheta'])
    
    # maybe apply selection on the number of post samples
    # for now, turn on and off by hand
    samples = post_samples[['theta', 'phi']].values
    sparse_samples = True
    if sparse_samples:
        samples = samples[::5]
    
    # make another loglike and like map, but with posterior samples overplotted
    hp.mollview(pixels['loglike'].values, title='NS null log likelihood + posterior samples', 
                cbar=True, cmap='inferno')
    # plot every 10th posterior sample (there are many)
    for sample in samples:
        hp.projplot(*sample, 'b.', alpha=0.2)
    fig = plt.gcf()
    fig.savefig(join(fig_path, 'loglike_post_samples.pdf'))
    
    hp.mollview(10**pixels['loglike'].values, title='NS null likelihood + posterior samples', 
                cbar=True, cmap='inferno')
    # plot every 10th posterior sample
    for sample in samples:
        hp.projplot(*sample, 'w.', alpha=0.2)
    fig = plt.gcf()
    fig.savefig(join(fig_path, 'like_post_samples.pdf'))
    
    