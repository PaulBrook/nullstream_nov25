#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:31:07 2020

@author: jgoldstein

NB These methods don't work very well, use ligo.skymap things instead (in albertos_pp)

make a plot of number of pulsars vs X% posterior area
- for each number of pulsars N, load posterior samples and posterior kde 
    for P (full FD likelihood) and Q (null-stream null only likelihood)
- compute the areas, by way of healpy pixelation, that contain X% of the 
    total posterior for some values of X (50%, 90% and 95%)
- make a plot of N vs area, for each X and for both P and Q
- plot healpy maps of the posterior KDE within each X% area, per N (and P/Q)
"""

import pickle
import numpy as np
import pandas as pd
import healpy as hp

from os.path import join
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

from pp_utils import get_post


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


def post_contour_area(skykde, nside=16, contour=0.9):
    """
    Compute smallest sky area that encloses X part of the posterior, with 
    X some value between 0 and 1. For example, if X = 0.9, it computes the area
    within the 90% posterior probability contour.
    """
    # grid sky into equal area pixels with healpy
    npixels = hp.nside2npix(nside)
    hp_pixels = np.arange(npixels)
    # area of one pixel from total sky area of 4pi sterradians
    # converted to square degrees
    pix_area = 4*np.pi * (180/np.pi)**2 / npixels
    
    # make pandas dataframe with row for each pixel, with theta and phi columns
    theta, phi = hp.pix2ang(nside, hp_pixels)
    # adjust coordinate system (ours has phi from -pi to pi, with phi=0 in the center,
    # healpy convention is phi from 0 to 2pi, with phi=pi in the center)
    phi = phi - np.pi
    pixels = pd.DataFrame(data={'theta':theta, 'phi':phi}, index=hp_pixels)
    
    # add costheta column so we can put (costheta, phi) into skykde
    # then add the skykde value as a columns also
    pixels['costheta'] = np.cos(pixels['theta'])
    pixels['skykde'] = skykde(pixels[['costheta', 'phi']].values.T)
    
    # to get the smallest number of pixels summing to 90% of the total posterior
    # we want to sort by desceding poster (i.e. skykde) value
    pixels.sort_values(by='skykde', ascending=False, inplace=True)
    # then get the cumulative value
    pixels['cum_skykde'] = np.cumsum(pixels['skykde'])
    # to find the 90% treshold, normalize by the total posterior (which will be
    # the last cumulative value)
    pixels['cum_skykde_norm'] = pixels['cum_skykde'] / pixels['cum_skykde'].iloc[-1]
    
    # find the treshold: first pixel that makes cumulative posterior go over 0.9
    treshold_pixel = np.searchsorted(pixels['cum_skykde_norm'], contour, side='right')
    # area to make 90% of the posterior, including the pixel that makes it go over 90%
    contour_area = pix_area * (treshold_pixel + 1)
    
    return contour_area, pixels, treshold_pixel


def plot_contour_area(pixels, treshold_pixel, nside=16, contour=0.9, 
                      save_path='//home/jgoldstein/Documents/projects/ptacake_runs/sky_runs/contour_skykde.pdf'):
    
    # select the pixels within the contour based on treshold pixel (inclusive)
    # then get the pixel numbers of these (the index)
    pixels_in_contour = pixels.iloc[0:treshold_pixel+1].index.values
    
    # make healpy map
    # fill all values with hp.UNSEEN to mask them
    hp_map = np.full((hp.nside2npix(nside)), hp.UNSEEN)
    # fill in all the pixels in the contour with the skykde (posterior) value
    #hp_map[pixels_in_contour] = pixels['skykde'].loc[pixels_in_contour]
    for p in range(hp.nside2npix(nside)):
        if p in pixels_in_contour:
            hp_map[p] = pixels['skykde'].loc[p]
            
    # healpy convention is centered around a different point then ours
    # but we can plot the map rotated with the desired center pixel being (theta=pi/2, phi=pi)
    # to find (lon, lat) which the rotator requires, convert to pix and back

    hp_center_pix = hp.ang2pix(nside, np.pi/2, np.pi)
    hp_center_lonlat = hp.pix2ang(nside, hp_center_pix, lonlat=True)
    
    hp_img = hp.mollview(hp_map, rot=(*hp_center_lonlat, 0),
                         title='Posterior up to {}%'.format(int(100*contour)))
    fig = plt.gcf()
    fig.savefig(save_path)
    
    
    # doesn't actually make sense I think
def post_area_costheta(skykde, ngrid=1e4, contour=0.9):
    """
    Compute area on which posterior (from skykde) sums to X part of the total posterior,
    with X given by the contour parameter. Use a grid in cos(theta), phi
    rather than a healpy sky grid. We can integrate over the sky spherical 
    surface by integrating over d cos(theta), d phi.
    However, in this method the pixels aren't equal area. So we have to compute
    cumulative integrated values (rather than just cumulative sums) which will
    give us the set of pixels that form the posterior contour area. Then we can
    compute the size of the area with another integral (integrate over the value 1).
    """
    
    # build meshgrid of costheta, phi values
    costheta = np.linspace(-1.0, 1.0, num=ngrid)
    phi = np.linspace(0.0, 2*np.pi, num=ngrid)
    csmesh, phimesh = np.meshgrid(costheta, phi)
    
    # flatten meshes so we have can make a dataframe with a long list of (costheta, phi) pixels
    pixels = pd.DataFrame(data={'costheta':csmesh.flatten(), 'phi':phimesh.flatten()})
    
    # compute skykde (posterior) values at each pixel and sort by this
    pixels['skykde'] = skykde(pixels[['costheta', 'phi']].values.T)
    pixels.sort_values(by='skykde', ascending=False, inplace=True)
    
    # compute cumulative integral with cumtrapz
    
def post_area_fit_gaussian(skykde, contour=0.9):
    """
    Compute area on which posterior (from skykde) sums to X part of the total 
    posterior, with X given by the contour parameter. Estimate by fitting a 
    2D gaussian. This method could work for high N-pulsar, small, nearly-gaussian
    posterior blobs.
    """
    
    

Psource_dir = '/home/jgoldstein/Documents/projects/ptacake_runs/sky_runs/FD_nonoise/'
Qsource_dir = '/home/jgoldstein/Documents/projects/ptacake_runs/sky_runs/NSnull_nonoise/'
npulsars = [3, 5, 8, 10, 20, 30, 50] # 100 not done yet
contours = [0.5, 0.9, 0.95]
Pcontour_areas = np.zeros((len(npulsars), len(contours)))
Qcontour_areas = np.zeros((len(npulsars), len(contours)))

# we need a high enough healpy resolution such that there are enough pixels within each posterior
# contour area to not get too big an error on pixel rounding
# for higher number of pulsars, this means we need a higher nside, so try these:
nside_from_npulsar = {3:16, 5:32, 8:32, 10:64, 20:128, 30:128, 50:128}

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
            Ppost = get_post(Ppost_path)
            Pskykde = load_kde(Pskykde_path)
            Qskykde = load_kde(Qskykde_path)
        except FileNotFoundError:
            print('Could not find all files for N={}'.format(N))
            continue
        
        nside=nside_from_npulsar[N]
        for cidx, c in enumerate(contours):
            # 50% contour can be significantly smaller then others, so double resolution
            if c == 0.5:
                nside = 2*nside
            
            Parea, Ppixels, Ptreshold = post_contour_area(Pskykde, contour=c, nside=nside)
            Qarea, Qpixels, Qtreshold = post_contour_area(Qskykde, contour=c, nside=nside)
            Pcontour_areas[Nidx, cidx] = Parea
            Qcontour_areas[Nidx, cidx] = Qarea
            
            # if the number of pixels in the contour is too low, we may get a large error on the
            # contour area, so check that here
            if Ptreshold < 100:
                print('Too few ({}) pixels within {}% contour for P, N={}'.format(Ptreshold, int(c*100), N))
            if Qtreshold < 100:
                print('Too few ({}) pixels within {}% contour for Q, N={}'.format(Qtreshold, int(c*100), N))


#            # also plot a healpy map for each contour area
#            filename = 'post_contour{}%_map.pdf'.format(int(c*100))
#            Psave = join(Poutputdir, filename)
#            Qsave = join(Qoutputdir, filename)
#            plot_contour_area(Ppixels, Ptreshold, nside=nside, contour=c, save_path=Psave)
#            plot_contour_area(Qpixels, Qtreshold, nside=nside, contour=c, save_path=Qsave)
        

    # save posterior contour areas results
    with open(join(Psource_dir, 'post_areas.csv', 'w+')) as fP:
        np.savetxt(fP, Pcontour_areas, delimiter=',')
    with open(join(Qsource_dir, 'post_areas.csv', 'w+')) as fQ:
        np.savetxt(fQ, Pcontour_areas, delimiter=',')
            
                              

def plot_npulsar_postarea(Pcontour_areas, Qcontour_areas, contours, npulsars,
                          savepath='/home/jgoldstein/Documents/projects/'
                          'ptacake_runs/sky_runs/posterior_areas.pdf'):
    # make plot of posterior contour areas
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    colours = ['red', 'blue', 'black']
    for cidx, c in enumerate(contours):
        c_percent = int(c*100)
        ax.plot(npulsars, Pcontour_areas[:, cidx], c=colours[cidx], 
                linestyle='-', marker='o', label='P {}%'.format(c_percent))
        ax.plot(npulsars, Qcontour_areas[:, cidx], c=colours[cidx], 
                linestyle='--', marker='d', label='Q {}%'.format(c_percent))
    
    ax.set_yscale('log')
    ax.set_ylabel('sky area (sq. deg.) in posterior contour')
    ax.set_xscale('log')
    ax.set_xlabel('num pulsars')
    
    # manually set xticks and labels
    ax.set_xticks(npulsars)
    ax.set_xticklabels([str(n) for n in npulsars])
    ax.set_xlim(min(npulsars)*0.9, max(npulsars)*1.1)
    
    ax.legend(loc='best')
    fig.savefig(savepath)
    
