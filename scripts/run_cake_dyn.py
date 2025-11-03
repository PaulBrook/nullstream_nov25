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
import glob
import numpy as np
import sys
import time

# Add the directory containing the script to sys.path
sys.path.append('/rds/projects/v/vecchioa-gw-pta/brookp/clean/NullStreams_orig/ptacake')

# Remove previously loaded ptacake module
if 'ptacake' in sys.modules:
    del sys.modules['ptacake']

# Add the new directory to sys.path
sys.path.insert(0, '/rds/projects/v/vecchioa-gw-pta/brookp/clean/NullStreams_orig')

import ptacake
from dynesty import NestedSampler

print(ptacake.__file__)

# Record start time
start_time = time.time()

# Import the module
from from_jannasutils import radec_location_to_ang, radec_to_thetaphi, radec_reduced_to_thetaphi

### command line options for sim and run config ###

parser = argparse.ArgumentParser("Make a PTA sim, then run cpnest")
parser.add_argument('-s', '--sim_config', required=True, dest='sim_config',
                    help='config file for PTA simulation')
parser.add_argument('-r', '--run_config', required=True, dest='run_config',
                    help='config file for CPNest run')
parser.add_argument('-o', '--output_dir', required=True,
                    help='Output directory')


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
    
#outdir = run_config['output_path']
outdir = args.output_dir
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
    raise Exception("Skipping sim pickle loading")
    with open(sim_pickle_path, 'rb') as f:
        sim = pickle.load(f)
        print('resuming run with stored pta_sim in {}'.format(sim_pickle_path))
        
#except FileNotFoundError:
except Exception:
    # we do not have a sim pickle to resume from a previous run, 
    # so do the simulation
    sim = ptacake.PTA_sim()
    
    # pulsar stuff
    method = sim_config['pulsar_method']
    
    if method == 'random':
        pulsar_opts = sim_config['pulsar_opts']
        num_pulsars = pulsar_opts.pop('num_pulsars')
        sim.random_pulsars(num_pulsars, **sim_config['pulsar_opts'])
        
    #elif method == 'from_file':
    #    sim.pulsars_from_file(sim_config['pulsar_file'])
        
    #elif method == 'from_array':
    #    sim.set_pulsars(sim_config['pulsar_array'], sim_config['pulsar_rms'])
        
    #elif method == 'from_csv':
    #    num_pulsars = sim_config['pulsar_opts']['num_pulsars']
    #    sim.pulsars_from_csv(sim_config['pulsar_file'], nrows=num_pulsars)
        
    #else:
    #    raise ValueError('Could not create or load pulsars with method {}'.format(method))
        
    # times stuff
    if sim_config['times_evenly_sampled']:
        sim.evenly_sampled_times(**sim_config['times_es_opts'])
    else:
        sim.randomized_times(**sim_config['times_rd_opts'])
    
    # signal and noise stuff
    #if sim_config['model_name'] in ['sinusoid_TD', 'Sinusoid_TD']:
    #    from ptacake.GW_models import sinusoid_TD
    #    sim.inject_signal(sinusoid_TD, sim_config['true_source'], *sim_config['true_args'])
    #else:
    #    raise NotImplementedError('Model {} not yet implemented'.format(sim_config['model_name']))
    
    #if sim_config['white_noise']:
        
        # try to read scale for white noise from config file, if not in config
        # or Null, set to 1 (then it doens't do anything)
    #    try:
    #        scale = sim_config['noise_scale']
    #    except:
    #        scale = 1
    #    if scale is None:
    #        scale = 1
        
    #    sim.white_noise(seed=sim_config['noise_seed'], scale=scale)


    
    ### LOAD THE REAL RESIDUALS AND TOAS IN                                                         

    ######################LOAD THE RESIDUALS AND TOAS#####################################          
    res_path = '/rds/projects/v/vecchioa-gw-pta/brookp/clean/first_try_Nullstream/res_and_toas/'
    sim_path = '/rds/projects/v/vecchioa-gw-pta/brookp/clean/first_try_Nullstream/res_and_toas/sim/'

    #file_paths_residuals = sorted(glob.glob(os.path.join(res_path, '*mean_residuals.txt')))
    file_paths_residuals = sorted(glob.glob(os.path.join(sim_path, '*sim_residuals_reg.txt')))
    file_paths_toas = sorted(glob.glob(os.path.join(sim_path, '*sim_toas_reg.txt')))
    #file_paths_toas = sorted(glob.glob(os.path.join(res_path, '*grouped_toas.txt')))

    print(len(file_paths_residuals))
    print(len(file_paths_toas))

    #11 PSRS: J0030+0451, J0613-0200, J1012+5307, J1455-3330, J1640+2224, J1643-1224, J1713+0747, B1855, B1937+21, J2145-0750, J2317+1439
    #9 PSRS: J0030+0451, J0613-0200, J1012+5307, J1455-3330, J1640+2224, J1643-1224, J1713+0747, J2145-0750, J2317+1439
    #indices = [1,9,15,22,26,27,29,  43,  50,  59,65] # all
    #rms_values = np.array([0.251e-6, 0.168e-6, 0.298e-6, 0.735e-6, 0.200e-6, 0.898e-6, 0.095e-6, 0.330e-6, 0.104e-6, 0.644e-6, 0.345e-6]) # white noise values from https://iopscience.iop.org/article/10.3847/2041-8213/acda9a/pdf        
    #rms_values = np.array([0.856e-6, 0.749e-6, 0.925e-6, 0.735e-6, 0.200e-6, 2.335e-6, 0.201e-6, 0.829e-6, 5.774e-6, 0.799e-6, 0.345e-6]) # full noise values from https://iopscience.iop.org/article/10.3847/2041-8213/acda9a/pdf

    #indices = [1,9,15,22,26,27,29] # first 7
    indices = [1,9,15,22,26,27,29,59,65] # all except 1855 and 1937
    #indices = [1,9,15,22,27,29,59,65] # all except 1855 and 1937 and 1640
    #indices = [1,9,15,22,26,27,59,65] # all except 1855 and 1937 and 1713
    #indices = [1,9,15,22,27,59,65] # all except 1855 and 1937 and 1713 and 1640 
    #rms_values = np.array([0.856e-6, 0.749e-6, 0.925e-6, 0.735e-6, 0.200e-6, 2.335e-6, 0.201e-6]) # first 7
    rms_values = np.array([0.856e-6, 0.749e-6, 0.925e-6, 0.735e-6, 0.200e-6, 2.335e-6, 0.201e-6, 0.799e-6, 0.345e-6]) # all except 1855 and 1937
    #rms_values = np.array([0.856e-6, 0.749e-6, 0.925e-6, 0.735e-6, 2.335e-6, 0.201e-6, 0.799e-6, 0.345e-6]) # all except 1855 and 1937 and 1640
    #rms_values = np.array([0.856e-6, 0.749e-6, 0.925e-6, 0.735e-6, 0.200e-6, 2.335e-6, 0.799e-6, 0.345e-6]) # all except 1855 and 1937 and 1713
    #rms_values = np.array([0.856e-6, 0.749e-6, 0.925e-6, 0.735e-6, 2.335e-6, 0.799e-6, 0.345e-6]) # all except 1855 and 1937 and 1713 and 1640
    #rms_values = np.array([0.856e-7, 0.749e-7, 0.925e-7, 0.735e-7, 0.200e-7, 2.335e-7, 0.201e-7, 0.799e-7, 0.345e-7]) # made 10 times smaller. Works for flat residuals.

    #file_paths_residuals = [file_paths_residuals[i] for i in indices]
    #file_paths_toas = [file_paths_toas[i] for i in indices]

    print(f'residuals: {file_paths_residuals}')
    print(f'toas: {file_paths_toas}')

    for pulsars in file_paths_residuals:
        print(f'Pulsars included: {os.path.basename(pulsars)}')
        
    resids = []
    toas = []

    # Arrays to hold values                                                                         
    RAh = []
    RAm = []
    DECh = []
    DECam = []

    theta = []
    phi = []

    # Save the RA and DEC to arrays                                                                 
    for filename in file_paths_residuals:
        # Extract the base name of the file (without the directory and extension)                   
        base_name = os.path.basename(filename)

        # Extract the relevant part after the 'J'                                                   
        coords = base_name.split('_')[0][1:]  # This will give something like "0030+0451"           

        # Parse the values                                                                          
        rah = int(coords[0:2])   # First two digits for RAh                                         
        ram = int(coords[2:4])   # Next two digits for RAm                                          
        dech = int(coords[4:7])  # Next two including the negative for DECh                         
        decam = int(coords[7:9]) # Last two for DECam                                               

        # Append to respective arrays                                                               
        RAh.append(rah)
        RAm.append(ram)
        DECh.append(dech)
        DECam.append(decam)

        theta_val, phi_val = radec_location_to_ang([RAh[-1],RAm[-1],DECh[-1],DECam[-1]])

        theta.append(theta_val)
        phi.append(phi_val)

    theta = np.array(theta)
    phi = np.array(phi)

    #np.random.seed(42)  # Set seed for reproducibility
    #np.random.shuffle(theta)
    #np.random.shuffle(phi)    

    print(f'All thetas: {theta}')
    print(f'All phis: {phi}')    
    
    sim._pulsars['theta'] = theta
    sim._pulsars['phi'] = phi

    for i in range(len(file_paths_residuals)):
        resids.append(np.loadtxt(file_paths_residuals[i]))

        # Load TOAs and append to the toas list                                                     
        toas.append(np.loadtxt(file_paths_toas[i]))

    # Shuffle each residual array in-place
    #for each_pulsar in resids:
    #    np.random.shuffle(each_pulsar)
        
    # Determine the maximum length                                                                  
    max_length = max(len(res) for res in resids)

    # Create a 2D array filled with NaNs                                                            
    real_res = np.full((len(resids), max_length), np.nan)

    # Fill the 2D array with the 1D arrays                                                          
    for i, res in enumerate(resids):
        real_res[i, :len(res)] = res

    # Replace all non-NaN values with 0.0                                                           
    #real_res = np.where(np.isnan(real_res), real_res, 0.0)

    # Determine the maximum length                                                                  
    #max_length = max(len(res) for res in resids)                                                   

    # Create a 2D array filled with NaNs                                                            
    real_toas = np.full((len(toas), max_length), np.nan)

    # Fill the 2D array with the 1D arrays                                                          
    for i, t in enumerate(toas):
        real_toas[i, :len(t)] = t

    #print(f'MAX LEN {max_length}')                                                                 
    #print(f'shape of real toas {real_toas.shape}')                                                 

    sim.set_residuals(real_res, real_toas) #sets the signal to the real residuals and the noise to zero.






    # inject sinusoid signal
    if sim_config['model_name'] in ['sinusoid_TD', 'Sinusoid_TD']:
        print(f'IN THE SINUSOID PART')
        from ptacake.GW_models import sinusoid_TD
        # parameters past times are: phase, amplitude, polarization, cos(i), GW angular freq
        #sinusoid_args = [0.123, 2e-15, np.pi/7, 0.5, 2e-8]
        #sinusoid_args = [0.5, 1.0e-14, 0.5, 0.5, 2.0e-8]
        # choose source (theta, phi) coordinates
        #source = (0.8*np.pi, 1.3*np.pi)
        #source = (0.5, 0.5)
        #sim.inject_signal(sinusoid_TD, source, *sinusoid_args)
        #sim.inject_signal(sinusoid_TD, sim_config['true_source'], *sim_config['true_args'])
    else:
        print(f'NOT IN THE SINUSOID PART')
        raise NotImplementedError('Model {} not yet implemented'.format(sim_config['model_name']))
        
    # save time span for each pulsar                                                             
    sim._pulsars['T'] = np.nanmax(sim._times, axis=1) - np.nanmin(sim._times, axis=1)

    # save number of TOAs (not nan)                                                              
    sim._pulsars['nTOA'] = [np.sum(np.isfinite(t)) for t in sim._times]

    # set the rms values:                                                                        
    sim._pulsars['rms'] = rms_values

    # save the TD covariance matrix diag(sigma**2), and inverse per pulsar                       
    num_times = sim._pulsars['nTOA'].values
    #num_times = np.array([max_length, max_length, max_length, max_length], dtype=np.int64)      
    print('rms values',sim._pulsars['rms'].values)
    sigma2 = (sim._pulsars['rms'].values)**2
    #sigma2 = np.array(631,631,631,631)**2                                                       
    sim._TD_covs = [sigma2[i] * np.eye(num_times[i]) for i in range(sim._n_pulsars)]
    sim._TD_inv_covs = [np.linalg.inv(cov) for cov in sim._TD_covs]


   
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
    #fig0, ax0 = sim.plot_pulsar_map(plot_point=(source))
    fig0.savefig(join(outdir, 'pulsar_map.pdf'))
if sim_config['plot_residuals_TD']:
    fig1, ax1 = sim.plot_residuals()
    fig1.savefig(join(outdir, 'TDresiduals.pdf'))
if sim_config['plot_residuals_FD'] and 'FD' in run_config['ll_name']:
    fig2, ax2 = sim.plot_residuals_FD()
    fig2.savefig(join(outdir, 'FDresiduals.pdf'))
    
    
### select sampler and run! ###
    
#if run_config['sampler'] == 'cpnest':
#    from ptacake.cpnest_stuff import run
    
#elif run_config['sampler'] == 'grid':
#    from ptacake.grid_sampler import run

#else:
#    raise ValueError('Unknown sampler {}'.format(run_config['sampler']))

from ptacake.cpnest_stuff_dyn import dynesty_run

print('Moving to run... \n')
# call sampler run with sim object, run_config and output directory
dynesty_run(sim, run_config, outdir=outdir)
#run(sim, run_config, outdir=outdir)

# Record end time                                                                                
end_time = time.time()

# Calculate elapsed time in seconds                                                              
elapsed_time = end_time - start_time

# Convert to minutes and hours                                                                   
elapsed_minutes = elapsed_time / 60
elapsed_hours = elapsed_time / 3600

# Print the elapsed time in seconds, minutes, and hours                                          
print(f"Code ran for {elapsed_time:.2f} seconds.")
print(f"Code ran for {elapsed_minutes:.2f} minutes.")
print(f"Code ran for {elapsed_hours:.2f} hours.")
