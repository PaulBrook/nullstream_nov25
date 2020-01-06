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
    
    return contour_area
    

Psource_dir = '/home/jgoldstein/Documents/projects/ptacake_runs/sky_runs/FD_nonoise/'
Qsource_dir = '/home/jgoldstein/Documents/projects/ptacake_runs/sky_runs/NSnull_nonoise/'
npulsars = [3, 5, 10, 20, 30, 50] # 100 not done yet
KLDs = []
contours = [0.5, 0.9, 0.95]
Pcontour_areas = np.zeros((len(npulsars), len(contours)))
Qcontour_areas = np.zeros((len(npulsars), len(contours)))

if __name__ == '__main__':
    for Nidx, N in enumerate(npulsars):
        subdir = '{}pulsars'.format(N)
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
        
        Ppost_90area = post_contour_area(Pskykde, contour=0.9)
        Qpost_90area = post_contour_area(Qskykde, contour=0.9)
        print('For {} pulsars, P 90% posterior area is {}, for Q it is {}'.format(N, Ppost_90area, Qpost_90area))
        
        for cidx, c in enumerate(contours):
            Parea = post_contour_area(Pskykde, contour=c)
            Qarea = post_contour_area(Qskykde, contour=c)
            Pcontour_areas[Nidx, cidx] = Parea
            Qcontour_areas[Nidx, cidx] = Qarea
    
    # make plot of posterior contour areas
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for cidx, c in enumerate(contours):
        c_percent = int(c*100)
        ax.plot(Pcontour_areas[:, cidx], label='P {}%'.format(c_percent))
        ax.plot(Qcontour_areas[:, cidx], label='Q {}%'.format(c_percent))
    
    ax.set_yscale('log')
    ax.set_ylabel('sky area (sq. deg.) in posterior contour')
    ax.set_xlabel('num pulsars')
    
    # set xticklabels to number of pulsars used
    xticks = ax.get_xticks()
    xticks_ints = [int(i)%len(npulsars) for i in xticks]
    xtick_labels = [str(npulsars[i]) for i in xticks_ints]
    ax.set_xticklabels(xtick_labels)
    
    ax.legend(loc='best')
    fig.savefig('/home/jgoldstein/Documents/projects/ptacake_runs/sky_runs/posterior_areas.pdf')
    