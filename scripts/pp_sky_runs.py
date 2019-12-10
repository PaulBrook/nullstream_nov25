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

##### command line options ###
parser = argparse.ArgumentParser("Make a PTA sim, then run cpnest")
parser.add_argument('-s', '--sim_config', required=True, dest='sim_config',
                    help='config file for PTA simulation')
parser.add_argument('-r', '--run_config', required=True, dest='run_config',
                    help='config file for CPNest run')
parser.add_argument('-d', '--cpnest_output_dir', default='./', dest='cpnest_dir',
                    help='directory with output files from CPNest to use')
parser.add_argument('-o', '--output_dir', default='./output/', dest='out_dir',
                    help='directory to save plot in')
parser.add_argument('-p', '--plots', action='store_true', dest='plots',
                    help='make healpy map and zoomed plot of skykde')
parser.add_argument('-x', '--example', action='store_true', dest='example',
                    help='load example data rather than new posterior (for faster testing)')
args = parser.parse_args()

#args_placeholder = namedtuple('args_placeholder', ['example', 'plots'])
#args = args_placeholder(example=True, plots=True)


### make healpy map from skykde ####
def healpy_plot(skykde, pta_sim, true_source, outdir, nside=30):
    npix = hp.nside2npix(nside)
    pix = np.arange(npix)
    skymap = np.zeros_like(pix)
    
    for p in pix:
        theta, phi = hp.pix2ang(nside, p)
        costheta = np.cos(theta)
        skymap[p] = skykde([costheta, phi])
    
    # make healpy map, plot pulsar locations and true source location
    hp.mollview(skymap, title='KDE skymap')
    marker_sizes = (pta_sim._pulsars['rms'].values/1.e-7)**(-0.4)*10
    for p, pulsar in enumerate(pta_sim._pulsars[['theta', 'phi']].values):
        hp.projplot(*pulsar, marker='*', c='w', ms=marker_sizes[p])
    hp.projplot(*true_source, marker='+', c='k', ms=10)
    
    # save the figure
    fig = plt.gcf()
    fig.savefig(join(outdir, 'kde_skymap.pdf'))

    
### plot skykde on zoomed in plot ###
#ct = cos(theta)
def zoom_plot(skykde, true_source, outdir, npoints=500):
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    
    ## choose points for cs and phi and evaluate skykde ##
    cspoints = np.linspace(-1., 1., num=npoints)
    csindex = np.arange(npoints)
    # because we know our injected source is near phi = 0, we "cheat" and choose a 
    # theta range from -pi to pi (it's cyclical anyway)
    phipoints = np.linspace(-np.pi, np.pi, num=2*npoints)
    phiindex = np.arange(2*npoints)
    
    csmesh, phimesh = np.meshgrid(cspoints, phipoints)
    skyplot = np.array([skykde(point) for point in zip(csmesh, phimesh)])
    
    ## "zoom" in on skyplot by selecting no-zero part ##
    # select the smallest rectangle that doesn't exclude any rows/columns that sum to more than 1e-14
    
    # first collapse phi dimension to find min and max costheta index
    skyplot_onlycs = np.sum(skyplot, axis=0)
    nonzero_csindex = csindex[skyplot_onlycs > 1e-14]
    min_csindex = min(nonzero_csindex)
    max_csindex = max(nonzero_csindex)
    
    # then collapse costheta dimension and do the same
    skyplot_onlyphi = np.sum(skyplot, axis=1)
    nonzero_phiindex = phiindex[skyplot_onlyphi > 1e-16]
    min_phiindex = min(nonzero_phiindex)
    max_phiidex = max(nonzero_phiindex)
    
    # select zoomed in parts of cs, phi and skyplot meshgrids
    #zoom = (slice(min_csindex, max_csindex+1), slice(min_phiindex, max_phiidex+1))
    zoom = (slice(min_phiindex, max_phiidex+1), slice(min_csindex, max_csindex+1))
    csmesh_zoom = csmesh[zoom]
    phimesh_zoom = phimesh[zoom]
    skyplot_zoom = skyplot[zoom]
    
    ax2.pcolormesh(csmesh_zoom, phimesh_zoom, skyplot_zoom, linewidth=0,rasterized=True)
    
    ## plot lines at true values ##
    ax2.vlines(np.cos(true_source[0]), *ax2.get_ylim(), linestyle='--', color='k')
    ax2.hlines(true_source[1], *ax2.get_xlim(), linestyle='--', color='k')
    
    ## ticklabels, axislabels etc ##
    ax2.set_xlabel(r'$\cos(\theta)$')
    ax2.set_ylabel(r'$\phi$')
    
    phiticks = ax2.get_yticks()
    phi_tlbs = [r'${:.2f}\pi$'.format(t/np.pi) for t in phiticks]
    ax2.set_yticklabels(phi_tlbs)
    
    csticks = ax2.get_xticks()
    cs_tlbs = [f'{t:.2f}' for t in csticks]
    ax2.set_xticklabels(cs_tlbs)
    
    fig2.savefig(join(outdir, 'kde_skyplot.png'), dpi=800)
    
    

if __name__ == "__main__":

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
    
    ### read in pta_sim from pickle ###
    
    pta_sim_path = join(cpnest_path, 'pta_sim.pickle')
    with open(pta_sim_path, 'rb') as f4:
        pta_sim = pickle.load(f4)
    
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
    
    # try to load skykde form pickle
    # will fail if it hasn't been made yet, so then make it and save it
    
    try:
        with open(join(outdir, 'skykde.pickle'), 'rb') as f2:
            skykde = pickle.load(f2)
    except FileNotFoundError:
        skykde = gaussian_kde(post[['costheta', 'phi']].values.T)
        with open(join(outdir, 'skykde.pickle'), 'wb+') as f3:
            pickle.dump(skykde, f3)
    
    ### optional plots ###
    if args.plots:
        healpy_plot(skykde, pta_sim, true_source, outdir, nside=32)
        zoom_plot(skykde, true_source, outdir, npoints=500)
    
    

    

