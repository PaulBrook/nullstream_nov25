#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:21:32 2019

@author: jgoldstein

post-process cpnest results from sky runs
get theta, phi posteriors
...
"""

from os.path import join, isfile
import sys
import argparse
import yaml
import pickle
from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import healpy as hp

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


# here is the example directory
# use this for faster testing of this script
example_dir = '/home/jgoldstein/Documents/projects/ptacake_runs/sky_runs/example'

#### command line options ###
#parser = argparse.ArgumentParser("Make a PTA sim, then run cpnest")
#parser.add_argument('-s', '--sim_config', required=True, dest='sim_config',
#                    help='config file for PTA simulation')
#parser.add_argument('-r', '--run_config', required=True, dest='run_config',
#                    help='config file for CPNest run')
#parser.add_argument('-d', '--cpnest_output_dir', default='./', dest='cpnest_dir',
#                    help='directory with output files from CPNest to use')
#parser.add_argument('-o', '--output_dir', default='./output/', dest='out_dir',
#                    help='directory to save plot in')
#parser.add_argument('-x', '--example', action='store_true', dest='example',
#                    help='load example data rather than new posterior (for faster testing)')
#args = parser.parse_args()

args_placeholder = namedtuple('args_placeholder', 'example')
args = args_placeholder(example=True)

if args.example:
    # if example, set paths from example dir
    sim_config_path = join(example_dir, 'sim_config.yaml')
    run_config_path = join(example_dir, 'run_config.yaml')
    cpnest_path = join(example_dir, 'output')
    outdir = join(example_dir, 'output')
else:
    # check that sim_config and run_config exist
    if not isfile(args.sim_config):
        parser.error('sim config file {} does not exist!'.format(args.sim_config))
    if not isfile(args.run_config):
        parser.error('run config file {} does not exist!'.format(args.run_config))
        
    sim_config_path = args.sim_config
    run_config_path = args.run_config
    cpnest_path = args.cpnest_dir
    outdir = args.out_dir
    
### read in true param values from sim config ###

with open(sim_config_path, 'r') as f1:
    sim_config = yaml.safe_load(f1)
true_source = sim_config['true_source']

### read in posterior file ###
        
post_file = join(cpnest_path, 'posterior.dat')
if not isfile(post_file):
    raise FileNotFoundError('Can not find posterior.dat in {}'.format(cpnest_path))
    
# does not read header with column names, because that line starts with #
post = pd.read_csv(post_file, delim_whitespace=True, header=None, comment='#')
# read column names from first line of posterior.dat
with open(post_file) as f3:
    first_line = f3.readline()
cols = first_line.split()[1:]
post.columns = cols

### make KDE from sky posterior ###

# if example, try to load skykde from pickle
if args.example:
    try:
        with open(join(example_dir, 'skykde.pickle'), 'rb') as f2:
            skykde = pickle.load(f2)
    except FileNotFoundError:            
        # otherwise make it, then save it
        skykde = gaussian_kde(post[['costheta', 'phi']].values.T)
        with open(join(example_dir, 'skykde.pickle'), 'wb+') as f3:
            pickle.dump(skykde, f3)
else:
    skykde = gaussian_kde(post[['costheta', 'phi']].values.T)

### make healpy map from skykde ####

nside = 30
npix = hp.nside2npix(nside)
pix = np.arange(npix)
skymap = np.zeros_like(pix)

for p in pix:
    theta, phi = hp.pix2ang(nside, p)
    costheta = np.cos(theta)
    skymap[p] = skykde([costheta, phi])
    
# read in pta_sim from pickle (so we can plot the pulsars)
pta_sim_path = join(cpnest_path, 'pta_sim.pickle')
with open(pta_sim_path, 'rb') as f4:
    sim = pickle.load(f4)
    
# get true source location from sim_config
true_source = sim_config['true_source']

# make healpy map, plot pulsar locations and true source location
hp.mollview(skymap, title='KDE skymap')
marker_sizes = (sim._pulsars['rms'].values/1.e-7)**(-0.4)*10
for p, pulsar in enumerate(sim._pulsars[['theta', 'phi']].values):
    hp.projplot(*pulsar, marker='*', c='w', ms=marker_sizes[p])
hp.projplot(*true_source, marker='+', c='k', ms=10)

# save the figure
fig = plt.gcf()
fig.savefig(join(outdir, 'kde_skymap.pdf'))







### get kde from sky posterior ###
# adapted from Albertos code http://gitlab.sr.bham.ac.uk/LISADA/bham-codes/blob/master/albertos/postprocessing.py #
#
#theta_post = post['theta'].values
#phi_post = post['phi'].values
#
## convert to longitude, latitude, and put into astropy coordinates
#lon_post = phi_post
#lat_post = theta_post - np.pi/2
#coordinates = SkyCoord(lon_post, lat_post, unit="rad")
#
## compute ligo skymap KDE from posterior points
#points = np.column_stack((coordinates.ra.radian, coordinates.dec.radian))
## if example, read in skyKDE from pickle, because this step takes looong
#if args.example:
#    with open(join(example_dir, 'example_skyKDE.pickle', 'r')) as f4:
#        skyKDE = pickle.load(f4)
#else:
#    skyKDE = Clustered2DSkyKDE(points, jobs=1)
#    
#    
#### code adapted from Albertos to make a skymap from the skyKDE ###
#    
## using Albertos default values
#top_nside = 16
#multi_order = True    
#    
## Construct the healpix map
#hpmap = skyKDE.as_healpix(top_nside=top_nside)
#if not multi_order:
#    hpmap = rasterize(hpmap)   
#    
#
## Save the map
#if multi_order:
#    fits_name = "hpmap_multiorder.fits"
#else:
#    fits_name = "hpmap.fits"
#fits_path = join(args.out_dir, fits_name)
#
#io.write_sky_map(fits_path, hpmap, nest=True)



