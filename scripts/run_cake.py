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
import pickle

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
    
    
### read in sim config and run config ###
    
with open(args.sim_config, 'r') as f1:
    sim_config = yaml.safe_load(f1)
    
with open(args.run_config, 'r') as f2:
    run_config = yaml.safe_load(f2)
    
    
### get and adjust output path ###
    
outdir = run_config['output_path']
# check if environment variable TMPDIR exists and if run_config option to use
# it is True. If so, put output dir inside there.
if 'TMPDIR' in os.environ and run_config['use_tmp']:
    tmpdir = os.environ['TMPDIR']
    if os.path.exists(tmpdir):
        print('Found TMPDIR (putting output dir within) {}'.format(tmpdir))
        outdir = join(tmpdir, outdir)
    
# normpath removes excess '/./'
outdir = os.path.normpath(outdir)
print('Putting plot etc output in {}'.format(outdir))

if not os.path.exists(outdir):
    os.mkdir(outdir)
    print('created dir {}. succes? {}'.format(outdir, os.path.exists(outdir)))    


### make or load PTA sim ###
    
# first check for pta sim pickle (resume from previous run)  
sim_pickle_path = join(outdir, 'pta_sim.pickle')
try:
    with open(sim_pickle_path, 'rb') as f:
        sim = pickle.load(f)
        print('resuming run with stored pta_sim in {}'.format(sim_pickle_path))
        
except FileNotFoundError:
    # we do not have a sim pickle to resume from a previous run, 
    # so do the simulation
    sim = ptacake.PTA_sim()
    
    # pulsar stuff
    method = sim_config['pulsar_method']
    
    if method == 'random':
        pulsar_opts = sim_config['pulsar_opts']
        num_pulsars = pulsar_opts.pop('num_pulsars')
        sim.random_pulsars(num_pulsars, **sim_config['pulsar_opts'])
        
    elif method == 'from_file':
        sim.pulsars_from_file(sim_config['pulsar_file'])
        
    elif method == 'from_array':
        sim.set_pulsars(sim_config['pulsar_array'], sim_config['pulsar_rms'])
        
    elif method == 'from_csv':
        num_pulsars = sim_config['pulsar_opts']['num_pulsars']
        sim.pulsars_from_csv(sim_config['pulsar_file'], nrows=num_pulsars)
        
    else:
        raise ValueError('Could not create or load pulsars with method {}'.format(method))
        
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
        
        
    # if using FD likelihood, need to run fourier_residuals
    # if using FD_ns likelihood, need to run concatenate_residuals also
    if 'FD' in run_config['ll_name']:
        sim.fourier_residuals()
    # null stream likelihoods need concatenated residuals
    if run_config['ll_name'] in ['FD_ns', 'FD_null']:
        sim.concatenate_residuals()
        
    # save the sim as a pickle in case we resume later
    with open(sim_pickle_path, 'wb') as f:
        pickle.dump(sim, f)
        
    
### optional plotting/save S/N ###
    
# compute and save S/N
snr = sim.compute_snr()
with open (join(outdir, 'snr.txt'), 'w+') as f:
    f.write('snr {}\n'.format(snr))

# plotting and saving plots
if sim_config['plot_pulsar_map']:
    fig0, ax0 = sim.plot_pulsar_map(plot_point=sim_config['true_source'])
    fig0.savefig(join(outdir, 'pulsar_map.pdf'))
if sim_config['plot_residuals_TD']:
    fig1, ax1 = sim.plot_residuals()
    fig1.savefig(join(outdir, 'TDresiduals.pdf'))
if sim_config['plot_residuals_FD'] and 'FD' in run_config['ll_name']:
    fig2, ax2 = sim.plot_residuals_FD()
    fig2.savefig(join(outdir, 'FDresiduals.pdf'))
    
    
### select sampler and run! ###
    
if run_config['sampler'] == 'cpnest':
    from ptacake.cpnest_stuff import run
    
elif run_config['sampler'] == 'grid':
    from ptacake.grid_sampler import run

else:
    raise ValueError('Unknown sampler {}'.format(run_config['sampler']))

print('Moving to run... \n')
# call sampler run with sim object, run_config and output directory
run(sim, run_config, outdir=outdir)

    
