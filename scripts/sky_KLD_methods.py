#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:26:28 2019

@author: jgoldstein

kl divergence between sky posteriors
"""
from os.path import join
import sys
import pickle
import yaml
import numpy as np
import pandas as pd
import healpy as hp

import matplotlib.pyplot as plt


# try one case, comparing the sky posteriors from two 10 pulsar runs (FD and NSnull)
path1 = '/home/jgoldstein/Documents/projects/ptacake_runs/sky_runs/FD_nonoise/10pulsars/output'
path2 = '/home/jgoldstein/Documents/projects/ptacake_runs/sky_runs/NSnull_nonoise/10pulsars/output'

with open(join(path1, 'skykde.pickle'), 'rb') as f1:
    skykde1 = pickle.load(f1)
with open(join(path2, 'skykde.pickle'), 'rb') as f2:
    skykde2 = pickle.load(f2)
    
def method1():
    # get a set of points to compute the kl divergence over from healpy
    nside = 64
    npix = hp.nside2npix(nside)
    pix = np.arange(npix)
    theta, phi = hp.pix2ang(nside, pix)
    points = np.array([np.cos(theta), phi])
    
    # get values from KDEs
    P = skykde1(points)
    Q = skykde2(points)
    
    # get rid of points where P or Q is too small
    too_small = (P < 1e-16) | (Q < 1e-16)
    good_pix = pix[~too_small]
    goodP = P[good_pix]
    goodQ = Q[good_pix]
    
    # normalise
    goodP_norm = goodP / np.sum(goodP)
    goodQ_norm = goodQ / np.sum(goodQ)
    
    # compute KL div
    KLD = np.sum(goodP_norm * np.log(goodP_norm/goodQ_norm))
    print('KL Divergence of Q from P with method 1: {} nats'.format(KLD))
    
    fig = plt.figure(figsize=(5, 8))
 
    theta, phi = hp.pix2ang(nside, good_pix)
    costheta = np.cos(theta)
    
    ax1 = fig.add_subplot(311)
    ax1.hexbin(costheta, phi, goodP_norm)
    ax1.set_title('P')
    
    ax2 = fig.add_subplot(312)
    ax2.hexbin(costheta, phi, goodQ_norm)
    ax2.set_title('Q')
    
    ax3 = fig.add_subplot(313)
    ax3.hexbin(costheta, phi, goodP_norm/goodQ_norm)
    ax3.set_title('P/Q')
    
    fig.tight_layout()
    return fig


# read in both posterior files
def read_post_file(post_file):
    post = pd.read_csv(post_file, delim_whitespace=True, header=None, comment='#')
    # read column names from first line of posterior.dat
    with open(post_file) as f3:
        first_line = f3.readline()
    cols = first_line.split()[1:]
    post.columns = cols
    return post

## different method, from posterior samples ##
def method2():

    post1 = read_post_file(join(path1, 'posterior.dat'))
    #post2 = read_post_file(join(path2, 'posterior.dat'))
    
    # compute KL div as a monte-carlo integral over the posterior points of post1
    # so we need to compute log (post1 / post2) at each of those points
    # for consistency, use skykde to compute both of those
    
    post1_points = post1[['costheta', 'phi']].values.T
    # compute values of P and Q KDEs at points of posterior 1 (P)
    P_at1 = skykde1(post1_points)
    Q_at1 = skykde2(post1_points)
    
    bad_points = np.arange(len(P_at1))[(P_at1 < 1e-16) | (Q_at1 < 1e-16)]
    print('removing bad points (zeros) {}'.format(bad_points))
    
    # get rid of zeros
    good = (P_at1 > 1e-16) & (Q_at1 > 1e-16)
    goodP_at1 = P_at1[good]
    goodQ_at1 = Q_at1[good]
    
    # normalise
    P_at1_norm = goodP_at1 / np.sum(goodP_at1)
    Q_at1_norm = goodQ_at1 / np.sum(goodQ_at1)
    
    # for monte-carlo integral, sum over values of integrand at the monte-carlo points 
    # (the points of posterior 1). Divide by the number of points
    n_points = np.sum(good)
    KLD2 = np.sum(np.log(P_at1_norm/ Q_at1_norm)) / n_points
    print('KL Divergence of Q from P with method 2: {} nats'.format(KLD2))
    
    fig = plt.figure(figsize=(5, 8))
 
    costheta, phi = post1_points
    costheta = costheta[good]
    phi = phi[good]
    
    ax1 = fig.add_subplot(311)
    ax1.hexbin(costheta, phi, P_at1_norm)
    ax1.set_title('P')
    
    ax1.set_xlim(-0.15, 0.35)
    ax1.set_ylim(0.0, 0.65)
    
    ax2 = fig.add_subplot(312, sharex=ax1, sharey=ax1)
    ax2.hexbin(costheta, phi, Q_at1_norm)
    ax2.set_title('Q')
    
    ax3 = fig.add_subplot(313, sharex=ax1, sharey=ax1)
    ax3.hexbin(costheta, phi, P_at1_norm/Q_at1_norm)
    ax3.set_title('P/Q')
    
    fig.tight_layout()
    
    return P_at1_norm, Q_at1_norm, KLD2, fig


## method 3: integrate with posterior samples but compute likelihood ratio ##

# the ratio of likelihood is the same as the ratio of posteriors, since the 
# priors are equal. We don't need the KDE for the likelihoods

path1b = '/home/jgoldstein/Documents/projects/ptacake_runs/sky_runs/FD_nonoise/10pulsars/'
path2b = '/home/jgoldstein/Documents/projects/ptacake_runs/sky_runs/NSnull_nonoise/10pulsars/'

from ptacake.GW_models import sinusoid_TD
def eval_FD_like(pta_sim, point):
    source = [np.arccos(point['costheta']), point['phi']]
    # phase amp pol cosi GW_freq
    args = [point['phase'], 10**point['logamp'], point['pol'], point['cosi'], point['GW_freq']]
    loglike = pta_sim.log_likelihood_FD(source, sinusoid_TD, args, add_norm=False)
    return 10**loglike

def eval_NSnull_like(pta_sim, point):
    source = [np.arccos(point['costheta']), point['phi']]
    loglike = pta_sim.log_likelihood_FD_onlynull(source, add_norm=False)
    return 10**loglike

def method3():
    
    post1 = read_post_file(join(path1, 'posterior.dat'))
    post2 = read_post_file(join(path2, 'posterior.dat'))
    
    # load run configs to get likelihood names 
    # (although we will want the first one to be FD likelihood and the second one
    # to be NS null likelihood)
    with open(join(path1b, 'run_config.yaml'), 'r') as f5:
        run_config1 = yaml.safe_load(f5)
    ll1_name = run_config1['ll_name']
    with open(join(path2b, 'run_config.yaml'), 'r') as f6:
        run_config2 = yaml.safe_load(f6)
    ll2_name = run_config2['ll_name']
    
    # just assume this
    assert(ll1_name == 'FD')
    assert(ll2_name == 'FD_null')    
    
    # load pta sims
    with open(join(path1, 'pta_sim.pickle'), 'rb') as f7:
        pta_sim1 = pickle.load(f7)
    with open(join(path2, 'pta_sim.pickle'), 'rb') as f8:
        pta_sim2 = pickle.load(f8)
    
    # evaluate P likelihood at P posterior points
    # evaluate Q likelihood at P posterior points
    # first check for pickles (because this takes a long time to do)
    npoints = len(post1)
    pickle_path = '/home/jgoldstein/Documents/projects/ptacake_runs/sky_runs/KLD_example'
    try:
        with open(join(pickle_path, 'Pll_at1.pickle'), 'rb') as f1:
            Plike_at1 = pickle.load(f1)
        with open(join(pickle_path, 'Qll_at1.pickle'), 'rb') as f2:
            Qlike_at1 = pickle.load(f2)
            
    except FileNotFoundError:
        Plike_at1 = np.array([eval_FD_like(pta_sim1, post1.iloc[p]) for p in range(npoints)])
        Qlike_at1 = np.array([eval_NSnull_like(pta_sim2, post1.iloc[p]) for p in range(npoints)])
        with open(join(pickle_path, 'Pll_at1.pickle'), 'wb+') as f3:
            pickle.dump(Plike_at1, f3)
        with open(join(pickle_path, 'Qll_at1.pickle'), 'wb+') as f4:
            pickle.dump(Qlike_at1, f4)
         
    #normalise
    Plike_norm = Plike_at1 / np.sum(Plike_at1)
    Qlike_norm = Qlike_at1 / np.sum(Qlike_at1)
    
    # compute KL Div
    KLD3 = np.sum(np.log(Plike_norm / Qlike_norm)) / npoints
    print('KL Divergence of P from Q with method 3: {} nats'.format(KLD3))
    
    fig = plt.figure(figsize=(5, 8))
 
    costheta = post1['costheta'].values
    phi = post1['phi'].values
    
    ax1 = fig.add_subplot(311)
    ax1.hexbin(costheta, phi, Plike_norm, gridsize=(100, 200))
    ax1.set_title('P')
    ax1.set_ylim(0.0, 0.65)
    
    ax2 = fig.add_subplot(312)
    ax2.hexbin(costheta, phi, Qlike_norm, gridsize=(100, 200))
    ax2.set_title('Q')
    ax2.set_ylim(0.0, 0.65)
    
    ax3 = fig.add_subplot(313)
    ax3.hexbin(costheta, phi, Plike_norm/Qlike_norm, gridsize=(100, 200))
    ax3.set_title('P/Q')
    ax3.set_ylim(0.0, 0.65)
    
    fig.tight_layout()
    
    return Plike_norm, Qlike_norm, KLD3, fig


# check normalisation of KDE (assuming scipy noramlises this but worth checking)
# ---> not normalised (because depends on total input points and stuff), so always normalize
# check normalisation of likelhiood
# compare KDE of the posterior with the (noramlised) likelihood
# they should be the same apart form a constant factor (prior or normalisation or both)
# --> likelihood has very high peak near the end of the posterior samples chain, which 
#     posterior from KDE has not. Likelihood of the other distribution also does NOT
#     have this peak, so that makes a big difference in the KDE (also indirectly via the normalisation
#     of the likelihood, since it affects all the values).
    
# hypothesis for why plots of method3 look crap: points that we have thrown out 
# for method2 fuck everything up in hexbin








    