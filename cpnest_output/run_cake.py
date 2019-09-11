#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 12:55:14 2019

@author: jgoldstein
"""
import yaml
import os
from os.path import join, isfile
import argparse

import ptacake

### command line options for sim and run config ###
parser = argparse.ArgumentParser("Make a PTA sim, then run cpnest")
parser.add_argument('-s', '--sim_config', required=True, dest='sim_config',
                    help='config file for PTA simulation')
parser.add_argument('-r', '--run_config', required=True, dest='run_config',
                    help='config file for CPNest run')
args = parser.parse_args()

# check that both config files exist
if not isfile(args.sim_config):
    parser.error('sim config file {} does not exist!'.format(args.sim_config))
if not isfile(args.run_config):
    parser.error('run config file {} does not exist!'.format(args.run_config))
    
### read in sim config and make PTA sim ###
    
with open(args.sim_config, 'r') as f1:
    sim_config = yaml.safe_load(f1)

sim = ptacake.PTA_sim()

# pulsar stuff
if sim_config['pulsar_method'] == 'random':
    pulsar_opts = sim_config['pulsar_opts']
    num_pulsars = pulsar_opts.pop('num_pulsars')
    sim.random_pulsars(num_pulsars, **sim_config['pulsar_opts'])
    
elif sim_config['pulsar_method'] == 'from_file':
    sim.pulsars_from_file(sim_config['pulsar_file'])
    
elif sim_config['pulsar_method'] == 'from_array':
    sim.set_pulsars(sim_config['pulsar_array'], sim_config['pulsar_rms'])
    

# times stuff
if sim_config['times_evenly_sampled']:
    sim.evenly_sampled_times(**sim_config['times_es_opts'])
else:
    sim.randomized_times(**sim_config['times_rd_opts'])

# signal and noise stuff
if sim_config['model_name'] in ['sinusoid_TD', 'Sinusoid_TD']:
    from ptacake.GW_models import sinusoid_TD
    sim.inject_signal(sinusoid_TD, sim_config['true_source'], *sim_config['true_args'])
else:
    raise NotImplementedError('Model {} not yet implemented'.format(sim_config['model_name']))

if sim_config['white_noise']:
    sim.white_noise()
    

### read run config, prepare for run ###

with open(args.run_config, 'r') as f2:
    run_config = yaml.safe_load(f2)
    
# if using FD likelihood, need to run fourier_residuals
# if using FD_ns likelihood, need to run concatenate_residuals also
if 'FD' in run_config['ll_name']:
    sim.fourier_residuals()
if run_config['ll_name'] == 'FD_ns':
    sim.concatenate_residuals()
    
### optinal plotting ###
outdir = run_config['output_path']
if not os.path.exists(outdir):
    os.mkdir(outdir)
    print('created dir {}. succes? {}'.format(outdir, os.path.exists(outdir)))
    
# compute and save S/N
snr = sim.compute_snr()
with open (join(outdir, 'snr.txt'), 'w') as f:
    f.write('snr {}\n'.format(snr))

if sim_config['plot_pulsar_map']:
    fig0_maybelist = sim.plot_pulsar_map(plot_point=sim_config['true_source'])
    try:
        fig0 = fig0_maybelist[0]
    except:
        fig0 = fig0_maybelist
    fig0.savefig(join(outdir, 'pulsar_map.pdf'))
if sim_config['plot_residuals_TD']:
    fig1 = sim.plot_residuals()
    fig1.savefig(join(outdir, 'TDresiduals.pdf'))
if sim_config['plot_residuals_FD'] and 'FD' in run_config['ll_name']:
    fig2 = sim.plot_residuals_FD()
    fig2.savefig(join(outdir, 'FDresiduals.pdf'))
    
### select sampler and run! ###
    
if run_config['sampler'] == 'cpnest':
    from ptacake.cpnest_stuff import run
    
elif run_config['sampler'] == 'grid':
    from ptacake.grid_sampler import run

else:
    raise ValueError('Unknown sampler {}'.format(run_config['sampler']))

run(sim, run_config)

    
