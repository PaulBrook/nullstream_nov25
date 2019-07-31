#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:50:53 2019

@author: jgoldstein
"""

import timeit

res = timeit.timeit(setup="""
import numpy as np
from scipy.linalg import block_diag
import random
import ptacake
import matplotlib.pyplot as plt

sim = ptacake.PTA_sim()
sim.random_pulsars(10, sig_rms=5e-8, uniform=False)
sim.randomized_times(t_end = 10*ptacake.YEAR, mean_cadence=2e6)

from ptacake.GW_models import sinusoid_TD
source = (np.pi*0.5, np.pi*0.1)
sinusoid_args = [0.123, 1e-14, np.pi/7, 0.5, 2e-8]
sim.inject_signal(sinusoid_TD, source, *sinusoid_args)

sim.fourier_residuals()
sim.concatenate_residuals()

print('P={}, N={}'.format(sim._n_pulsars, sim._n_freqs))

from ptacake.nullstream_algebra import construct_M
off_source = (np.pi*0.8, np.pi*1.4)
ns_mat = construct_M(*off_source, sim._pulsars[['theta', 'phi']].values)
small_ns_mat = ns_mat

P = sim._n_pulsars
N = sim._n_freqs

big_ns_mat = np.zeros((P*N, P*N))
for j in range(N):
    # starting from point [j, j], put the elements of the small ns_mat
    # at every point [j + a*N, j + b*N], with a and b any integers.
    big_ns_mat[j::N, j::N] = small_ns_mat
""",


stmt="""
M, Zinv, logdetZ = sim._ns_covariance(ns_mat)
""", 

number=10)

print('\n timeit result \n{}'.format(res))
