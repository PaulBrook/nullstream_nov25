#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:02:07 2019

@author: jgoldstein
"""

import numpy as np
import ptacake as cake
import matplotlib.pyplot as plt

seed = 1003

sim = cake.PTA_sim()
# random pulsars with very low rms noise level
sim.random_pulsars(4, mean_rms=1e-8, sig_rms=5e-9, seed=seed)
fig1, ax1 = sim.plot_pulsar_map()
fig1.savefig('./pulsar_map.pdf')

mean_cadence = 1000000.0
# randomize some pulsar start times
start_times = np.random.rand(4) * 100 * mean_cadence
sim.randomized_times(mean_cadence=mean_cadence, t_start=start_times, gaps=True, seed=seed)

# inject sinusoid signal
from ptacake.GW_models import sinusoid_TD
# parameters past times are: phase, amplitude, polarization, cos(i), GW angular freq
sinusoid_args = [0.123, 1e-14, np.pi/7, 0.5, 2e-8]
# choose source (theta, phi) coordinates
source = (0.8*np.pi, 1.3*np.pi)
sim.inject_signal(sinusoid_TD, source, *sinusoid_args)

# inject white noise at previously determined residuals rms level
sim.white_noise()

fig2, ax2 = sim.plot_residuals()
fig2.savefig('./TD_residuals.pdf')

sim.fourier_residuals()
fig3, ax3 = sim.plot_residuals_FD()
fig3.savefig('./FD_residuals.pdf')

# null-stream stuff
sim.concatenate_residuals()
ns_mat = cake.nullstream_algebra.construct_M(*source, sim._pulsars[['theta', 'phi']].values)
big_ns_mat, inv_ns_cov = sim._ns_covariance(ns_mat)
null_streams = big_ns_mat @ sim.residuals_concat

# split up null_streams (concatenated) into separate null-streams
ns_hp = null_streams[:sim._n_freqs]
ns_hc = null_streams[sim._n_freqs:2*sim._n_freqs]
ns1 = null_streams[2*sim._n_freqs:3*sim._n_freqs]
ns2 = null_streams[3*sim._n_freqs:]

fig4, ax4 = plt.subplots(1)
ax4.plot(sim._freqs, ns_hp, label='$h^{+}$')
ax4.plot(sim._freqs, ns_hc, label=r'$h^{\times}$')
ax4.plot(sim._freqs, ns1, label='null-stream 1')
ax4.plot(sim._freqs, ns2, label='null-stream 2')
ax4.legend()
ax4.set_xlabel('Frequency (Hz)')
fig4.savefig('./null_streams_FD.pdf')

ns_hp_TD = np.fft.irfft(ns_hp)
ns_hc_TD = np.fft.irfft(ns_hc)
ns1 = np.fft.irfft(ns1)
ns2 = np.fft.irfft(ns2)

fig5, ax5 = plt.subplots(1)
ax5.plot(ns_hp_TD, label='$h^{+}$')
ax5.plot(ns_hc_TD, label=r'$h^{\times}$')
ax5.plot(ns1, label='null-stream 1')
ax5.plot(ns2, label='null-stream 2')
ax5.legend()
ax5.set_xlabel('Time (steps)')
fig5.savefig('./null_streams_TD.pdf')


