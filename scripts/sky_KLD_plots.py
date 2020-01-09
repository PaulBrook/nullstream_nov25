#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 13:08:19 2019

@author: jgoldstein
"""

import pickle
import numpy as np
import pandas as pd
import healpy as hp

from os.path import join
import matplotlib.pyplot as plt


def read_post_file(file_path):
    post = pd.read_csv(file_path, delim_whitespace=True, header=None, comment='#')
    # read in column names from first line
    # (can't read them in with pandas bc they're commented with #)
    with open(file_path) as f:
        first_line = f.readline()
    cols = first_line.split()[1:]
    post.columns = cols
    return post

def load_kde(file_path):
    with open(file_path, 'rb') as f:
        skykde = pickle.load(f)
    return skykde

def kde_at_post(skykde, post):
    """
    skykde: gaussian_kde on (costheta, phi)
    post: pandas dataframe with posterior samples, includes costheta and phi columnns
    """
    # evaluate skykde at posterior (costheta, phi) points
    post_points = post[['costheta', 'phi']].values
    skykde_values = skykde(post_points.T)
    
    # normalise skykde_values
    skykde_norm = skykde_values / np.sum(skykde_values)
    
    return skykde_norm

def compute_KLD(Ppost, Pskykde, Qskykde):
    # KLD(P | Q) = int( P * log(P/Q) )
    # compute int (P ...) by summing over P posterior points
    # compute log(P/Q) with values from kde (evaluated at P posterior points)
    
    Pkde_at_P = kde_at_post(Pskykde, Ppost)
    Qkde_at_P = kde_at_post(Qskykde, Ppost)
    
    # replace zero values with really small number (so we don't get zero division errors)
    Pkde_at_P[Pkde_at_P < 1e-16] = 1e-16
    Qkde_at_P[Qkde_at_P < 1e-16] = 1e-16
    
    npoints = len(Ppost)
    KLD = (1/npoints) * np.sum( np.log(Pkde_at_P / Qkde_at_P) )
    return KLD
    

Psource_dir = '/home/jgoldstein/Documents/projects/ptacake_runs/sky_runs/FD_nonoise/'
Qsource_dir = '/home/jgoldstein/Documents/projects/ptacake_runs/sky_runs/NSnull_nonoise/'
npulsars = [3, 5, 8, 10, 20, 30, 50] # 100 not done yet
KLDs = []


if __name__ == '__main__':
    for Nidx, N in enumerate(npulsars):
        subdir = '{}pulsars'.format(N)
        Poutputdir = join(Psource_dir, subdir, 'output')
        Qoutputdir = join(Qsource_dir, subdir, 'output')
        Ppost_path = join(Psource_dir, subdir, 'output', 'posterior.dat')
        Pskykde_path = join(Psource_dir, subdir, 'output', 'skykde.pickle')
        Qpost_path = join(Qsource_dir, subdir, 'output', 'posterior.dat')
        Qskykde_path = join(Qsource_dir, subdir, 'output', 'skykde.pickle')
        
        
        try:
            Ppost = read_post_file(Ppost_path)
            Pskykde = load_kde(Pskykde_path)
            Qskykde = load_kde(Qskykde_path)
        except FileNotFoundError:
            print('Could not find all files for N={}'.format(N))
            continue
        
        KLD = compute_KLD(Ppost, Pskykde, Qskykde)
        KLDs.append(KLD)
        print('For {} pulsars, KL-Divergence between P (FD like) and Q (NS null like) is: {}'.format(N, KLD))
        
                              

    