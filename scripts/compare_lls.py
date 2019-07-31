#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:57:19 2019

@author: jgoldstein
"""


import numpy as np
from scipy.linalg import block_diag
import ptacake
import matplotlib.pyplot as plt

sim = ptacake.PTA_sim()
sim.random_pulsars(5, sig_rms=5e-8, uniform=False)
sim.evenly_sampled_times(cadence=2e6, T=10*ptacake.YEAR)
#sim.randomized_times(t_end = 10*ptacake.YEAR, mean_cadence=2e6)

from ptacake.GW_models import sinusoid_TD
source = (np.pi*0.5, np.pi*0.1)
sinusoid_args = [0.123, 1e-14, np.pi/7, 0.5, 2e-8]
sim.inject_signal(sinusoid_TD, source, *sinusoid_args)

sim.fourier_residuals()
#sim.fft_residuals()
sim.concatenate_residuals()


# vary phase
phases = np.linspace(0, 2*np.pi, num=50)
ll_TD_phases = np.zeros(len(phases))
ll_FD_phases = np.zeros(len(phases))
ll_TD_ns_phases = np.zeros(len(phases))
ll_FD_ns_phases = np.zeros(len(phases))

for i, phase in enumerate(phases):
    test_args = sinusoid_args.copy()
    test_args[0] = phase
    ll_TD_phases[i] = sim.log_likelihood_TD(source, sinusoid_TD, test_args)
    ll_FD_phases[i] = sim.log_likelihood_FD(source, sinusoid_TD, test_args)
    ll_TD_ns_phases[i] = sim.log_likelihood_TD_ns(source, sinusoid_TD, test_args)
    ll_FD_ns_phases[i] = sim.log_likelihood_FD_ns(source, sinusoid_TD, test_args)
    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(phases, ll_TD_phases, linestyle='-', label='TD')
ax.plot(phases, ll_FD_phases, linestyle='--', label='FD')
ax.plot(phases, ll_TD_ns_phases, linestyle=':', label='TD ns')
ax.plot(phases, ll_FD_ns_phases, linestyle='-.', label='FD ns')

ax.set_xlabel('phase')
ax.set_ylabel('log likelihood')
ax.legend(loc='best')
fig.tight_layout()

fig.savefig('/home/jgoldstein/Documents/projects/Nullstream/compare_lls.pdf')