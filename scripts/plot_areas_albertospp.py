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

import matplotlib.pyplot as plt
from matplotlib import gridspec

Psource_dir = '/home/jgoldstein/Documents/projects/ptacake_runs/sky_runs/FD_nonoise/'
Qsource_dir = '/home/jgoldstein/Documents/projects/ptacake_runs/sky_runs/NSnull_nonoise/'
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
            
            Qdata = np.loadtxt(Qfile, skiprows=1)
            post_areas.loc[nump, 'Qarea'] = Qdata[:, 1]
            
        except OSError:
            print('Could not find (all) data for npulsars={}, skipping'.format(nump))
        
    return post_areas
        

def plot_post_areas(post_areas,
                    savepath='/home/jgoldstein/Documents/projects/'
                          'ptacake_runs/sky_runs/post_areas_albertospp.pdf'):
    
    # drop NaNs
    post_areas.dropna(inplace=True)
    npulsars_used = post_areas.index.get_level_values('npulsars').unique().values
    contours_used = post_areas.index.get_level_values('contour').unique().values
    
    # make a figure etc
    fig = plt.figure(figsize=(5, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    
    colours = ['red', 'blue', 'black']
    
    for cidx, c in enumerate(contours_used):
        
        # plot npulsars vs posterior area
        # separately for P and Q
        Pareas = post_areas.xs(c, level=1)['Parea'].values
        Qareas = post_areas.xs(c, level=1)['Qarea'].values
        
        c_percent = int(c*100)
        ax.plot(npulsars_used, Pareas, c=colours[cidx],
                linestyle='-', marker='o', label='P {}%'.format(c_percent))
        ax.plot(npulsars_used, Qareas, c=colours[cidx],
                linestyle='--', marker='d', label='Q {}%'.format(c_percent))
        
        # in the second plot, plot the relative difference between Q and P
        diff = (Qareas - Pareas)/Pareas
        print(diff)
        ax2.plot(npulsars_used, diff, c=colours[cidx], 
                 linestyle='-', marker='s', label='{}%'.format(c_percent))
        
    # x-axis stuff, npulsars
    ax.set_xscale('log')
    ax2.set_xlabel('Num. pulsars')
    
    # manually set xticks and labels
    ax.set_xticks(npulsars_used)
    ax.set_xticks([], minor=True) 
    ax.set_xticklabels([str(int(n)) for n in npulsars_used])
    ax.set_xlim(min(npulsars_used)*0.9, max(npulsars_used)*1.1)
    
    # plot reference line at 0 for difference
    ax2.hlines(0, *ax.get_xlim(), color='gray', linestyle='--')
    
    # set and label two y-axes
    ax.set_yscale('log')
    ax.set_ylabel('Sky area in posterior (sq. deg.)')
    ax.legend(loc='best')
    
    ax2.set_ylabel('Rel. difference')
    ax2.legend(loc='best')
    
    fig.tight_layout()
    fig.savefig(savepath)
    return fig
    

if __name__ == '__main__':
    post_areas = load_postareas(Psource_dir, Qsource_dir, contours, npulsars)
    fig = plot_post_areas(post_areas)
