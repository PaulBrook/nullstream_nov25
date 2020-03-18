#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:45:10 2020

@author: jgoldstein

plot contour areas vs number of pulsars
use pp results made with ligo.skymap and albertos postprocessing
"""

import numpy as np
import pandas as pd
from os.path import join
import os

import matplotlib.pyplot as plt
from matplotlib import gridspec

npulsars = [3, 5, 8, 10, 20, 30, 50, 100]
contours = [0.5, 0.9, 0.95]


def load_postareas(Psource_dir, Qsource_dir, contours, npulsars):
    multi_index = pd.MultiIndex.from_product([npulsars, contours], names=['npulsars', 'contour'])
    post_areas = pd.DataFrame(index=multi_index, columns=['Parea', 'Qarea'])
    
    for nump in npulsars:
        
        try:
            Pfile = join(Psource_dir, '{}pulsars'.format(nump), 'output', 'contour_areas.txt')
            Qfile = join(Qsource_dir, '{}pulsars'.format(nump), 'output', 'contour_areas.txt')
            
            # find posterior areas in second column of each data file and put into DataFrame
            Pdata = np.loadtxt(Pfile, skiprows=1)
            post_areas.loc[nump, 'Parea'] = Pdata[:, 1]
            
            try:
                Qdata = np.loadtxt(Qfile, skiprows=1)
                post_areas.loc[nump, 'Qarea'] = Qdata[:, 1]
            except:
                print('No Q for {} pulsars'.format(nump))
            
        except OSError:
            print('Could not find (all) data for npulsars={}, skipping'.format(nump))
        
    return post_areas

def get_postpart_stats(Psource_dir, Qsource_dir, contours, npulsars):
    
    multi_index = pd.MultiIndex.from_product([npulsars, contours], 
                                             names=['npulsars', 'contour'])
    postpart_stats = pd.DataFrame(index=multi_index, 
            columns=['Parea_mean', 'Parea_std', 'Qarea_mean', 'Qarea_std'])
    
    sources = {'P':Psource_dir, 'Q':Qsource_dir}
    
    for nump in npulsars:
        
        for key in sources:
        
            # get list of all contour areas files for skymap parts
            output_dir = join(sources[key], '{}pulsars'.format(nump), 'output')
            files_all = os.listdir(output_dir)
            files = [f for f in files_all if (f.find('part') > -1) and (f.find('contour_areas') > -1)]
            
            # for each part contour file, read in data, and collect contour areas
            areas = np.zeros(shape=(len(contours), len(files)))
            for f_idx, f in enumerate(files):
                data = np.loadtxt(join(output_dir, f), skiprows=1)
                areas[:, f_idx] = data[:, 1]
                
            # then save std and mean to dataframe
            postpart_stats.loc[nump, '{}area_mean'.format(key)] = areas.mean(axis=1)
            postpart_stats.loc[nump, '{}area_std'.format(key)] = areas.std(axis=1)
    
    return postpart_stats


def plot_post_areas(ax1, ax2, post_areas,
                    colours=['red', 'blue', 'black'], alpha=1, labels=True):
    # ax1 = main figure
    # ax2 = small figure for Q/P ratio
    
    # drop NaNs
    #post_areas.dropna(inplace=True)
    npulsars_used = post_areas.index.get_level_values('npulsars').unique().values
    contours_used = post_areas.index.get_level_values('contour').unique().values
    

    colours = ['red', 'blue', 'black']
    
    for cidx, c in enumerate(contours_used):
        
        # plot npulsars vs posterior area
        # separately for P and Q
        Pareas = post_areas.xs(c, level=1)['Parea']
        Qareas = post_areas.xs(c, level=1)['Qarea']
        
        c_percent = int(c*100)
        if labels:
            ax1.plot(npulsars_used, Pareas.values, c=colours[cidx], alpha=alpha,
                     linestyle='-', marker='o', label='P {}%'.format(c_percent))
            ax1.plot(npulsars_used, Qareas.values, c=colours[cidx], alpha=alpha,
                     linestyle='--', marker='d', label='Q {}%'.format(c_percent))
        else:
            ax1.plot(npulsars_used, Pareas.values, c=colours[cidx], alpha=alpha,
                     linestyle='-', marker='o')
            ax1.plot(npulsars_used, Qareas.values, c=colours[cidx], alpha=alpha,
                     linestyle='--', marker='d')
    
        # in the second plot, plot the relative difference between Q and P
        #diff = (Qareas.values - Pareas.values)/Pareas.values
        diff = Qareas.values / Pareas.values
        if labels:
            ax2.plot(npulsars_used, diff, c=colours[cidx], alpha=alpha,
                     linestyle='-', marker='s', label='{}%'.format(c_percent))
        else:
            ax2.plot(npulsars_used, diff, c=colours[cidx], alpha=alpha,
                     linestyle='-', marker='s')
            
    return npulsars_used

        
def ax_settings(ax1, ax2, npulsars_used):
    # x-axis stuff, npulsars
    ax1.set_xscale('log')
    ax2.set_xlabel('Num. pulsars')
    
    # manually set xticks and labels
    ax1.set_xticks(npulsars_used)
    ax1.set_xticks([], minor=True) 
    ax1.set_xticklabels([str(int(n)) for n in npulsars_used])
    ax1.set_xlim(min(npulsars_used)*0.9, max(npulsars_used)*1.1)
    
    # plot reference line at 1 for ratio
    ax2.hlines(1, *ax1.get_xlim(), color='gray', linestyle='--')
    
    # set and label two y-axes
    ax1.set_yscale('log')
    ax1.set_ylabel('Sky area in posterior (sq. deg.)')
    ax1.legend(loc='best')
    
    #ax2.set_yscale('log')
    ax2.set_ylabel('Area ratio (Q/P)')
    ax2.legend(loc='best')
    

    

if __name__ == '__main__':
    # make a figure etc
    fig = plt.figure(figsize=(5, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    # main figure for npulsars vs Parea and Qarea
    ax1 = fig.add_subplot(gs[0])
    # smaller second panel for ratio Qarea/Parea
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    
    
    runs_dir = '/home/jgoldstein/Documents/projects/ptacake_runs'
    #sky_runs = ['sky_runs', 'sky_runs2']
    sky_runs = ['sky_runs', 'sky_runs']
    alphas = [1, 0.3]
    overall_n_used = []
    
    for i, run in enumerate(sky_runs):
        if i == 0:
            Psource_dir = join(runs_dir, run, 'FD_nonoise')
            Qsource_dir = join(runs_dir, run, 'NSnull_nonoise')
        else:
            Psource_dir = join(runs_dir, run, 'FD_wnoise')
            Qsource_dir = join(runs_dir, run, 'NSnull_wnoise')
    
        post_areas = load_postareas(Psource_dir, Qsource_dir, contours, npulsars)
        npulsars_used = plot_post_areas(ax1, ax2, post_areas, alpha=alphas[i], labels=(i==0))
        overall_n_used += list(npulsars_used)
    
    ax_settings(ax1, ax2, np.unique(overall_n_used))

    fig.tight_layout()
    savepath='/home/jgoldstein/Documents/projects/ptacake_runs/sky_runs/post_areas_runs1_nonoise+wnoise.pdf'
    fig.savefig(savepath)