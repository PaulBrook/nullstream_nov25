#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 16:28:14 2020

@author: jgoldstein
"""
G = 6.674e-11
c = 299792458
msun = 1.988435e30
year = 365.25 * 24 * 3600
Hubble_time = 4.55e17

import numpy as np

def fdot(chirp_mass, frequency):
    return (95/5.) * np.pi**(8/3) * ((G * chirp_mass * msun) / c**3)**(5/3) * frequency**(11/3)

def freq(chirp_mass, time_to_coal):
    return (1/np.pi) * ((G * chirp_mass * msun) / c**3)**(-5/8.) * ((5 / 256.) * (1/ time_to_coal))**(3/8)

def HellingsDowns(angle):
    
    term = (1 - np.cos(angle)) / 2
    return term * np.log( term ) - (1/6)*term + (1/3)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    
    angles = np.linspace(0, np.pi, 1000)
    HD = HellingsDowns(angles)
    HD[0] = 1/3
    angles_deg = angles * (180 / np.pi)
    
    ax.plot(angles_deg, HD, c='k')
    
    ax.set_xlim((0, 180))
    ax.hlines(0, *ax.get_xlim(), linestyle='--', color='xkcd:steel blue')
    
    ax.set_xlabel(r'$\gamma_{ij}$', fontsize=12)
    ax.set_ylabel(r'$C_{ij}$', fontsize=12)
    
    fig.tight_layout()
    fig.savefig('/home/jgoldstein/Documents/projects/janna-thesis/figures/Hellings_Downs.pdf')