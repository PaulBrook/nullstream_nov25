#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:00:31 2019

@author: jgoldstein
shamelessly stolen from albertos code
"""
import numpy as np
from os.path import join, isfile
import argparse
import yaml
import pandas as pd

from ptacake.plots import seaborn_corner

### command line options ###
parser = argparse.ArgumentParser("Make a PTA sim, then run cpnest")
parser.add_argument('-s', '--sim_config', required=True, dest='sim_config',
                    help='config file for PTA simulation')
parser.add_argument('-r', '--run_config', required=True, dest='run_config',
                    help='config file for CPNest run')
parser.add_argument('-d', '--cpnest_output_dir', default='./', dest='cpnest_dir',
                    help='directory with output files from CPNest to use')
parser.add_argument('-o', '--output_dir', default='./output/', dest='out_dir',
                    help='directory to save plot in')
args = parser.parse_args()

# check that both config files exist
if not isfile(args.sim_config):
    parser.error('sim config file {} does not exist!'.format(args.sim_config))
if not isfile(args.run_config):
    parser.error('run config file {} does not exist!'.format(args.run_config))
    
    
### read in true param values from sim config ###
    
with open(args.sim_config, 'r') as f1:
    sim_config = yaml.safe_load(f1)
if sim_config['model_name'] == 'sinusoid_TD':
    theta, phi = sim_config['true_source']
    phase, amp, pol, cosi, GW_freq = sim_config['true_args']
    true_params = dict(
            theta=theta,
            phi=phi,
            phase=phase,
            amp=amp,
            logamp=np.log10(amp),
            pol=pol,
            cosi=cosi,
            GW_freq=GW_freq)
else:
    raise NotImplementedError('Have not implemented parameter names for model {}'.format(sim_config['model_name']))
    
    
### check which params are actually sampled (from run config) ###
    
with open(args.run_config, 'r') as f2:
    run_config = yaml.safe_load(f2)
    
sampled_params = []
fixed_params = {}
for param, p_or_v in run_config['prior_or_value'].items():
    try:
        lower, upper = p_or_v
        sampled_params.append(param)
    except:
        fixed_params[param] = p_or_v

### read in posterior file ###
        
post_file = join(args.cpnest_dir, 'posterior.dat')
if not isfile(post_file):
    raise FileNotFoundError('Can not find posterior.dat in {}'.format(args.cpnest_dir))
    
# does not read header with column names, because that line starts with #
post = pd.read_csv(post_file, delim_whitespace=True, header=None, comment='#')
# read column names from first line of posterior.dat
with open(post_file) as f3:
    first_line = f3.readline()
cols = first_line.split()[1:]
post.columns = cols

### plot corner ###

# bad labels for now
labels=dict()
for param in true_params.keys():
    labels[param] = str(param)

figsave = join(args.out_dir, 'my_corner.pdf')

seaborn_corner(post, true_params, labels, param_names=sampled_params,
               savefile=figsave)
