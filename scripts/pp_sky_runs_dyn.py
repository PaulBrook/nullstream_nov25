#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:21:32 2019

@author: jgoldstein

post-process cpnest results from sky runs
- get theta, phi posterior and estimate a gaussian kde from it (and save it)
- make a full healpy sky map with the posterior kde
- make a zoomed in cos(theta) vs phi plot of the posterior kde 
    (with optionally overplotted posterior samples)
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

from pp_utils import get_post
import os

import matplotlib.cm as cm
import matplotlib.colors as colors

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

# ### ORIGINAL CODE ###
# ### make healpy map from skykde ####
# def healpy_plot(skykde, pta_sim, true_source, outdir, nside=30):
#     npix = hp.nside2npix(nside)
#     pix = np.arange(npix)
#     skymap = np.zeros_like(pix, dtype=float)
    
#     #    for p in pix:
#     #        theta, phi = hp.pix2ang(nside, p)
#     #        costheta = np.cos(theta)
#     #        skymap[p] = skykde([costheta, phi])
#     #    # normalize the skykde values
#     #    skymap = skymap / np.sum(skymap)
    
#     for p in pix:
#         theta, phi = hp.pix2ang(nside, p)
#         costheta = np.cos(theta)
#         kde_value = skykde([costheta, phi])
#         skymap[p] = kde_value[0]
#     skymap = skymap / np.sum(skymap)
    
#     # make healpy map, plot pulsar locations and true source location
#     hp.mollview(skymap, title='KDE skymap', unit='posterior prob.')
#     marker_sizes = (pta_sim._pulsars['rms'].values/1.e-7)**(-0.4)*10
#     for p, pulsar in enumerate(pta_sim._pulsars[['theta', 'phi']].values):
#         #hp.projplot(*pulsar, marker='*', c='w', ms=marker_sizes[p])
#         hp.projplot(*pulsar, marker='*', c='w', ms=10)
#     hp.projplot(*true_source, marker='+', c='k', ms=10)
    
#     # save the figure
#     fig = plt.gcf()
#     fig.savefig(join(outdir, 'kde_skymap.pdf'))

# ### FLIPPING THE ORIENTATION TO MATCH THE MPTA PLOTS ###
# ### make healpy map from skykde ####
# def healpy_plot(skykde, pta_sim, true_source, outdir, nside=30):
#     npix = hp.nside2npix(nside)
#     pix = np.arange(npix)
#     skymap = np.zeros_like(pix, dtype=float)
    
#     #    for p in pix:
#     #        theta, phi = hp.pix2ang(nside, p)
#     #        costheta = np.cos(theta)
#     #        skymap[p] = skykde([costheta, phi])
#     #    # normalize the skykde values
#     #    skymap = skymap / np.sum(skymap)
    
#     for p in pix:
#         theta, phi = hp.pix2ang(nside, p)
#         costheta = np.cos(theta)
#         kde_value = skykde([costheta, phi])
#         skymap[p] = kde_value[0]
#     skymap = skymap / np.sum(skymap)

#     # --------- HORIZONTAL FLIP ---------
#     # Flip phi -> 2π - phi
#     theta, phi = hp.pix2ang(nside, pix)
#     phi_flipped = (2 * np.pi - phi) % (2 * np.pi)
#     flipped_pix = hp.ang2pix(nside, theta, phi_flipped)
    
#     flipped_skymap = np.zeros_like(skymap)
#     flipped_skymap[flipped_pix] = skymap
#     # -----------------------------------
    
#     # Plot the flipped map
#     hp.mollview(flipped_skymap, title='KDE skymap (flipped)', unit='posterior prob.',
#                 rot=(180, 0, 0))  # ensures RA=180 is centered (for left-to-right RA)

#     # Plot flipped pulsars
#     marker_sizes = (pta_sim._pulsars['rms'].values / 1.e-7) ** (-0.4) * 10
#     for p, (theta_p, phi_p) in enumerate(pta_sim._pulsars[['theta', 'phi']].values):
#         phi_p_flipped = (2 * np.pi - phi_p) % (2 * np.pi)
#         hp.projplot(theta_p, phi_p_flipped, marker='*', c='w', ms=marker_sizes[p])
    
#     # Plot flipped true source
#     theta_t, phi_t = true_source
#     phi_t_flipped = (2 * np.pi - phi_t) % (2 * np.pi)
#     hp.projplot(theta_t, phi_t_flipped, marker='+', c='k', ms=10)
        
#     # save the figure
#     fig = plt.gcf()
#     fig.savefig(join(outdir, 'kde_skymap.pdf'))


def healpy_plot(skykde, pta_sim, true_source, outdir, nside=30):
    npix = hp.nside2npix(nside)
    pix = np.arange(npix)
    skymap = np.zeros_like(pix, dtype=float)

    for p in pix:
        theta, phi = hp.pix2ang(nside, p)
        costheta = np.cos(theta)
        kde_value = skykde([costheta, phi])
        skymap[p] = kde_value[0]
    skymap = skymap / np.sum(skymap)

    # --------- HORIZONTAL FLIP ---------
    theta, phi = hp.pix2ang(nside, pix)
    phi_flipped = (2 * np.pi - phi) % (2 * np.pi)
    flipped_pix = hp.ang2pix(nside, theta, phi_flipped)

    flipped_skymap = np.zeros_like(skymap)
    flipped_skymap[flipped_pix] = skymap

    # Plot the flipped map
    hp.mollview(flipped_skymap, title='KDE skymap (flipped)', unit='posterior prob.',
                rot=(180, 0, 0))  # centers RA=180

    # Plot flipped pulsars
    marker_sizes = (pta_sim._pulsars['rms'].values / 1.e-7) ** (-0.4) * 10
    for p, (theta_p, phi_p) in enumerate(pta_sim._pulsars[['theta', 'phi']].values):
        phi_p_flipped = (2 * np.pi - phi_p) % (2 * np.pi)
        hp.projplot(theta_p, phi_p_flipped, marker='*', c='w', ms=marker_sizes[p])

    # Plot flipped true source
    theta_t, phi_t = true_source
    phi_t_flipped = (2 * np.pi - phi_t) % (2 * np.pi)
    hp.projplot(theta_t, phi_t_flipped, marker='+', c='k', ms=10)

    selected_points_file = os.path.join(args.cpnest_dir, 'selected_sky_locations.txt')
    
    # --- Load selected sky locations ---
    data = np.loadtxt(selected_points_file, delimiter=',', dtype=str, skiprows=1)
    thetas = data[:, 0].astype(float)
    phis = data[:, 1].astype(float)
    logLs = data[:, 2].astype(float)

    # Sort indices by descending logL
    sorted_indices = np.argsort(logLs)[::-1]

    # Normalize logLs to [0, 1] for colormap
    norm = colors.Normalize(vmin=np.min(logLs), vmax=np.max(logLs))
    cmap = cm.get_cmap('coolwarm')  # Red-to-blue colormap (you could also try 'plasma' or 'viridis')

    
    # Plot selected points with labels 1..6 on flipped map
    for rank, idx in enumerate(sorted_indices, start=1):
        theta_sel = thetas[idx]
        phi_sel = phis[idx]
        logL_val = logLs[idx]
        
        # Apply the same horizontal flip to phi
        phi_sel_flipped = (2 * np.pi - phi_sel) % (2 * np.pi)

        # Get color based on normalized logL
        color = cmap(norm(logL_val))

        # Plot the point with the color
        hp.projscatter(theta_sel, phi_sel_flipped, coord='C', lonlat=False,
                   marker='o', c=[color], s=25, alpha=0.9)
        
        # Plot marker
        # Plot marker (red circles)
        #hp.projscatter(theta_sel, phi_sel_flipped, coord='C', lonlat=False, marker='o', c='r', s=50, alpha=0.8)

        #hp.projplot(theta_sel, phi_sel_flipped, marker='o', c='r', ms=12, alpha=0.8)

        # Get 2D projection coordinates to place label text
        #x, y = hp.projplot(theta_sel, phi_sel_flipped, coord='C', lonlat=False, return_projected=True)
        #x, y = hp.projscatter(theta_sel, phi_sel_flipped, coord='C', lonlat=False, return_projected=True, marker=None)
        #x, y = hp.projplot(theta_sel, phi_sel_flipped, coord='C', lonlat=False)

        # Draw label number (simple text)
        #plt.text(x, y, str(rank), color='yellow', fontsize=14, fontweight='bold',
        #         ha='center', va='center')

    # Save figure
    fig = plt.gcf()
    fig.savefig(join(outdir, 'kde_skymap.pdf'))
    plt.close(fig)    
    
### plot skykde on zoomed in plot ###
#ct = cos(theta)
def zoom_plot(skykde, true_source, outdir, npoints=500, overplot_samples=None):
    
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
    # normalize skykde values
    skyplot = skyplot / np.sum(skyplot)
    
    ## "zoom" in on skyplot by selecting non-zero part ##
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
    
    im = ax2.pcolormesh(csmesh_zoom, phimesh_zoom, skyplot_zoom, linewidth=0,rasterized=True)
    cbar = fig2.colorbar(im, ax=ax2, )
    cbar.set_label('posterior prob.')
    
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
    
    # overplot posterior samples
    ax2.scatter(*overplot_samples, c='w', marker='.', alpha=0.2)
    
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
    print(f'true_source: {true_source}')
    
    ### read in pta_sim from pickle ###

    import sys
    sys.path.insert(0, '/rds/projects/v/vecchioa-gw-pta/brookp/clean/NullStreams_orig')
    
    pta_sim_path = join(cpnest_path, 'pta_sim.pickle')
    with open(pta_sim_path, 'rb') as f4:
        pta_sim = pickle.load(f4)
    
    ### read in posterior file ###

    posterior_samples_file = os.path.join(cpnest_path, 'posterior_samples.txt')
    logweights_file = os.path.join(cpnest_path, 'logweights.txt')
    output_file = os.path.join(cpnest_path, 'posterior.dat')

    # Load posterior samples (7 columns of parameters)
    posterior_samples = np.loadtxt(posterior_samples_file)

    # Load log weights (1 column of logL)
    logL = np.loadtxt(logweights_file)

    # Create a logPrior column of zeros (since the priors are flat)
    logPrior = np.zeros(logL.shape)

    # Append the logL and logPrior as the 8th and 9th columns
    combined_data = np.hstack((posterior_samples, logL.reshape(-1, 1), logPrior.reshape(-1, 1)))

    # Define the header
    header = 'phase logamp pol cosi GW_freq costheta phi rmsfactor amp_0 gamma_0 amp_1 gamma_1 amp_2 gamma_2 amp_3 gamma_3 amp_4 gamma_4 logL logPrior'

    # Save the combined data as posteriors.dat with header
    np.savetxt(output_file, combined_data, fmt='%.8e', header=header)

    print(f'Combined file saved to: {output_file}')
    
    post_file = join(cpnest_path, 'posterior.dat')
    if not isfile(post_file):
        raise FileNotFoundError('Can not find posterior.dat in {}'.format(cpnest_path))
        
    post = get_post(post_file)
    
    ### make KDE from sky posterior ###
    
    # try to load skykde form pickle
    # will fail if it hasn't been made yet, so then make it and save it
    
    try:
        with open(join(outdir, 'skykde.pickle'), 'rb') as f2:
            skykde = pickle.load(f2)
    except FileNotFoundError:
        skykde = gaussian_kde(post[['costheta', 'phi']].values.T)
        os.makedirs(outdir, exist_ok=True)
        with open(join(outdir, 'skykde.pickle'), 'wb+') as f3:
            pickle.dump(skykde, f3)
    
    ### optional plots ###
    if args.plots:
        healpy_plot(skykde, pta_sim, true_source, outdir, nside=32)
        
        # plot posterior samples over zoom kde plot
        # there are many so only plot every 10th sample
        overplot_samples = post[['costheta', 'phi']].values.T[:, ::10]
        zoom_plot(skykde, true_source, outdir, npoints=500, 
                  overplot_samples=overplot_samples)
    
    

    

