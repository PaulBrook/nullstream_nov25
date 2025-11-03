"""
Created on Mon Aug  5 12:49:59 2019

@author: jgoldstein
"""
import numpy as np
import os
from os.path import join

import cpnest
import cpnest.model
from dynesty import utils as dyutils

class cpnest_model(cpnest.model.Model):
    """
    Generic Model class for the sampling algorithm

    Parameters
    ----------
    log_like_func: func
        should be albertos.sampler_ln_likelihood

    names: list of strings
        names of all parameters that need to be sampled

    bounds: list of lists [lower bound, upper bound]
        prior ranges for the parameters

    like_args: list
        arguments for log_likelihood

    like_kwargs: dict
        Keyword arguments for log_likelihood


    Returns
    -------

    cls: class
        This Model class, containing data and kwargs

    livepoint: class
        contains all parameters, likelihoods, etc required for the sampling

    """
        
    def __init__(self, prior_or_value, PTA_sim, ll_name, add_norm,
                 model_func=None, model_names=[], model_kwargs={}):
        """
        prior_or_value: dict
            keys must be all parameter names in model_names, plus 'costheta', 'phi'
            for the source location. Each value must either be a list or tuple of 
            [lower, upper] bounds for the parameter if it is to be sampled,
            or a "true" value if it is to be fixed.
        PTA_sim: PTA_sim object
        ll_name: str 
            one of {'TD', 'TD_ns', 'FD', 'FD_ns'}
        add_norm: bool
            if True, call likelihood with add_norm=True (False otherwise)
        model_func: python function 
            (e.g. sinusoid_TD)
        model_names: list of str
            names of the parameters of model_func in order, leave out "times"
            (e.g. ['phase', 'amp', 'pol', 'cosi', 'GW_freq'] for sinusoid_TD)
        model_kwargs: dict
            keyword arguments passed on to model_func
        """
        ll_funcs = {'TD': PTA_sim.log_likelihood_TD,
                    'TD_ns': PTA_sim.log_likelihood_TD_ns,
                    'FD': PTA_sim.log_likelihood_FD,
                    'FD_dyn': PTA_sim.log_likelihood_FD_dyn,
                    'FD_ns':PTA_sim.log_likelihood_FD_ns,
                    'FD_null':PTA_sim.log_likelihood_FD_onlynull
                    }
        
        # things we need in the log_likelihood function
        self.ll_name = ll_name
        self.ll_func = ll_funcs[ll_name]
        self.add_norm = add_norm
        self.model_func = model_func
        self.model_names = model_names
        self.model_kwargs = model_kwargs
        
        ### go through the "priors_or_values" dict to see which params are sampled ###
        sample_names = []
        sample_bounds = []
        # the dict "current_values" will hold the "true" values for now, then
        # in the likelihood evaluation we will update it with livepoint
        self.current_values = {}
        # make another version of current_values that will hold values without any logs
        self.current_values_nolog = {}
        for key, value in prior_or_value.items():
            
            # check for "log"
            if key[:3] == 'log':
                use_log = True
                pname = key[3:]
            else:
                use_log = False
                pname = key

            print(f'PNAME: {pname} SELF MODEL NAMEs: {self.model_names}')
                
            # first check the parameter name (without "log") is a valid model parameter name
            if not pname in self.model_names + ['costheta', 'phi']:
                raise ValueError('Unknown model parameter {} given in prior dict'.format(pname))
                
            # check if iterable with lower, upper is given or fixed value
            try:
                lower, upper = value
                sample_names.append(key)
                sample_bounds.append([lower, upper])
                self.current_values.update({key:None})
                self.current_values_nolog.update({pname:None})
            except:
                self.current_values.update({key:value})
                if use_log:
                    self.current_values_nolog.update({pname:10**value})
                else:
                    self.current_values_nolog.update({pname:value})
                
        # check we have an entry for each model parameter and costheta, phi
        for p in self.model_names + ['costheta', 'phi']:
            if not (p in self.current_values or ('log' + p) in self.current_values):
                raise ValueError('Missing fixed value or prior for parameter {} in prior dict'.format(p))
            if not p in self.current_values_nolog:
                raise ValueError('Missing fixed value or prior for parameter {} in prior dict'.format(p))
                
        # things that cpnest uses
        self.names = sample_names
        self.bounds = sample_bounds
        

    # investigate time for this log_likelihood call, using FD likelihood
    # and sim from '/home/jgoldstein/Documents/projects/ptacake_runs/all_params_evenly_sampled3/FD2
    # NB: timing this function (by calling mod.log_likelihood) does not work, don't exactly
    # know why but when checking the code with ?? it's some placeholder method rather than
    # the actual function (which doesn't do anything)
    def log_likelihood(self, livepoint):
        """
        Evaluate the likelihood on a livepoint
        """
        # negligible time
        # 299 ns ± 6.41 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
        self.current_values.update(livepoint)
        
        # negligible time
        # 2.78 µs ± 35.9 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        # go through params and convert log into not log
        for key, value in self.current_values.items():
            if key[:3] == 'log':
                self.current_values_nolog.update({key[3:]:10**value})
            else:
                self.current_values_nolog.update({key:value})
        
        # negligible time
        # 149 ns ± 2.59 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
        # split off costheta and phi parameters from other model parameters
        # and take arccos of costheta to get theta
        source = [np.arccos(self.current_values['costheta']), self.current_values['phi']]    
        
        # for FD_null likelihood, don't have model and model_args etc
        if self.ll_name == 'FD_null':
            ll = self.ll_func(source)
        
        else:
            # get the rest of the model parameters in the correct order
            # then pass to likelihood function
            
            # negligible time
            # 590 ns ± 6.32 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
            model_args = [self.current_values_nolog[p] for p in self.model_names]
            
            # this apparently takes more time than the whole function???
            # 4.2 ms ± 60.8 µs per loop (mean ± std. dev. of 7 runs, 100 loops each) (FD)
            # 185 ms ± 6.16 ms per loop (mean ± std. dev. of 7 runs, 1 loop each) (FD_null)
            ll = self.ll_func(source, self.model_func, model_args, 
                              add_norm=self.add_norm, return_only_norm=False, 
                              **self.model_kwargs)

        return ll
    


def run(PTA_sim, config, outdir='./output'):
    
    # for FD_null likelihood, don't need GW model etc
    if config['ll_name'] == 'FD_null':
        # takes negligible time also in cade of FD_null likelihood
        mod = cpnest_model(config['prior_or_value'], PTA_sim, 
                           config['ll_name'], config['add_norm'])
    
    else:
        if config['model_name'] in ['sinusoid', 'Sinusoid', 'sinusoid_TD', 'Sinusoid_TD']:
            from .GW_models import sinusoid_TD
            model_func = sinusoid_TD
            model_names = ['phase', 'amp', 'pol', 'cosi', 'GW_freq']
        else:
            raise NotImplementedError('Other models than GW sinusoid currently not implemented')
        
        # takes neglibile time
        # 9.45 µs ± 66.2 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        mod = cpnest_model(config['prior_or_value'], PTA_sim, config['ll_name'], 
                           config['add_norm'], model_func=model_func, 
                           model_names=model_names)
        
    # save zero likelihood for computation of log bayes factor later
    # doesn't make sense for null-only likelihood anyway so skip that one
    if config['ll_name'] != 'FD_null':

        # assumign sinusoid_TD model
        #['phase', 'amp', 'pol', 'cosi', 'GW_freq']
        zero_amp_args = [0, 0, 0, 0.5, 1e-8]
        zero_logl = mod.ll_func([0, 0], model_func, zero_amp_args, 
                                add_norm=config['add_norm'], return_only_norm=False)
        #zero_logl = mod.ll_func([0, 0], model_func, zero_amp_args)
        save_path = join(outdir, 'zero_logl.txt')
        # need w+ for write (w) and make the file if it doesn't exist already (+)
        with open(save_path, 'w+') as f:
            f.write('{}\n'.format(zero_logl))
                
    sampler_opts = config['sampler_opts']
    
    # check if environment variable SLURM_NTASKS exists. If so, use that for nthreads
    if 'SLURM_NTASKS' in os.environ:
        nthreads = int(os.environ['SLURM_NTASKS'])
    # otherwise, use nthreads from config
    else:
        nthreads = sampler_opts['nthreads']
    
    # negligible time
    # 669 ns ± 16.1 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    #Instantiate the sampler
    CPNest_args = dict(usermodel=mod,
                        nlive=sampler_opts['nlive'],
                        maxmcmc=sampler_opts['nsteps'],
                        nthreads=nthreads,
                        verbose=sampler_opts['verbosity'],
                        output=outdir,
                        resume=sampler_opts['resume'])
    # this only works for cpnest version 0.9.8 (installed from source), so we
    # don't want to add this keyword argument if it's not needed
    if sampler_opts['ncheckpoint'] is not None:
        CPNest_args.update(n_periodic_checkpoint=sampler_opts['ncheckpoint'])
        
    # this takes a bit of time (but it's done only once per run so doesn't matter)
    # 47.7 ms ± 1.33 ms per loop (mean ± std. dev. of 7 runs, 10 loops each) (FD likelihood)
    # 49.6 ms ± 2.1 ms per loop (mean ± std. dev. of 7 runs, 10 loops each) (FD_null likelihood)
    cpn = cpnest.CPNest(**CPNest_args)
    
    print('Putting cpnest output in {}'.format(cpn.output))
    print('Running CPNest!\n')
    cpn.run()
    
    # save the posterior samples
    cpn.get_nested_samples()
    cpn.get_posterior_samples()


from dynesty import NestedSampler

def dynesty_run(PTA_sim, config, outdir='./output'):

    # for FD_null likelihood, don't need GW model etc
    if config['ll_name'] == 'FD_null':
        # takes negligible time also in cade of FD_null likelihood
        mod = cpnest_model(config['prior_or_value'], PTA_sim,
                           config['ll_name'], config['add_norm'])

    else:
        if config['model_name'] in ['sinusoid', 'Sinusoid', 'sinusoid_TD', 'Sinusoid_TD']:
            from .GW_models import sinusoid_TD
            model_func = sinusoid_TD
            model_names = (['phase', 'amp', 'pol', 'cosi', 'GW_freq', 'rms_factor'] + [x for i in range(PTA_sim._n_pulsars) for x in (f'A_red_p{i}', f'gamma_red_p{i}')])

        else:
            raise NotImplementedError('Other models than GW sinusoid currently not implemented')

        # takes neglibile time
        # 9.45 µs ± 66.2 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        mod = cpnest_model(config['prior_or_value'], PTA_sim, config['ll_name'],
                           config['add_norm'], model_func=model_func,
                           model_names=model_names)

    # save zero likelihood for computation of log bayes factor later
    # doesn't make sense for null-only likelihood anyway so skip that one
    if config['ll_name'] != 'FD_null':
        # assumign sinusoid_TD model

        # Define true parameters from simulation
        true_params = np.array([
            0.5,           # phase (radians)
            1.0e-12,       # h (GW strain)
            0.5,           # psi (polarization)
            0.5,           # cosi (cos(inclination))
            3.0e-8,        # fgw (Hz)
            np.cos(1.5),   # cos_theta (theta = 1.5 rad)
            3.78,          # phi (radians)
            1.0,           # rms_factor
            #3.16227766e-13, 4.0,  # Pulsar 1: A_red, gamma
            #3.16227766e-13, 4.0,  # Pulsar 2
            #3.16227766e-13, 4.0,  # Pulsar 3
            #3.16227766e-13, 4.0,  # Pulsar 4
            #3.16227766e-13, 4.0   # Pulsar 5
            1.0e-13, 3.0,  # Pulsar 1: A_red, gamma
            1.0e-13, 3.0,  # Pulsar 2
            1.0e-13, 3.0,  # Pulsar 3
            1.0e-13, 3.0,  # Pulsar 4
            1.0e-13, 3.0   # Pulsar 5
        ])

        # Test parameters with offset GW amplitude
        test_params_gw = true_params.copy()
        test_params_gw[1] = 1.5e-12  # Offset h to 1.5e-12
        
        # Test parameters with red noise at prior boundaries
        test_params_red = true_params.copy()
        test_params_red[8] = 3.16e-13  # A_red = 10^(-12.5) for pulsar 1
        test_params_red[9] = 4.0       # gamma = 4.0 for pulsar 1
        
        # Compute likelihoods
        print("Testing likelihood with true parameters:")
        true_logl = mod.ll_func(true_params)
        print(f"True log-likelihood: {true_logl}")
        
        print("Testing likelihood with offset GW amplitude:")
        gw_logl = mod.ll_func(test_params_gw)
        print(f"Offset GW amplitude log-likelihood: {gw_logl}")
        
        print("Testing likelihood with offset red noise:")
        red_logl = mod.ll_func(test_params_red)
        print(f"Offset red noise log-likelihood: {red_logl}")
        

        #['phase', 'amp', 'pol', 'cosi', 'GW_freq']
        #zero_amp_args = [0, 0, 0, 0.5, 1e-8, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        zero_amp_args = [0, 0, 0, 0.5, 1e-8, 0, 0, 1] + [0] * (PTA_sim._n_pulsars * 2)
        #zero_logl = mod.ll_func([0, 0], model_func, zero_amp_args,
        #                        add_norm=config['add_norm'], return_only_norm=False)
        #zero_logl = mod.ll_func([0, 0], model_func, zero_amp_args)
        zero_logl = mod.ll_func(zero_amp_args)
        save_path = join(outdir, 'zero_logl.txt')
        # need w+ for write (w) and make the file if it doesn't exist already (+)
        with open(save_path, 'w+') as f:
            f.write('{}\n'.format(zero_logl))

    sys.exit()

            
    sampler_opts = config['sampler_opts']
    
    # Set thread count based on SLURM or config
    nthreads = int(os.environ.get('SLURM_NTASKS', sampler_opts['nthreads']))

    def prior_transform(utheta):
        transformed_params = []

        # Define the priors based on the config['prior_or_value']
        prior_ranges = config['prior_or_value']

        # Transform phase (uniform from 0 to 2π)
        phase_min, phase_max = prior_ranges['phase']
        transformed_params.append(phase_min + utheta[0] * (phase_max - phase_min))

        # Transform logamp (uniform from -16 to -12)
        logamp_min, logamp_max = prior_ranges['logamp']
        logamp = logamp_min + utheta[1] * (logamp_max - logamp_min)
        amplitude = 10 ** logamp
        transformed_params.append(amplitude)

        # Transform pol (uniform from 0 to π/4)
        pol_min, pol_max = prior_ranges['pol']
        transformed_params.append(pol_min + utheta[2] * (pol_max - pol_min))

        # Transform cosi (uniform from -1 to 1)
        cosi_min, cosi_max = prior_ranges['cosi']
        transformed_params.append(cosi_min + utheta[3] * (cosi_max - cosi_min))

        # Transform GW_freq (uniform from 0 to 1e-7)
        GW_freq_min, GW_freq_max = prior_ranges['GW_freq']
        #transformed_params.append(GW_freq_min + utheta[4] * (GW_freq_max - GW_freq_min))
        log_min = np.log10(GW_freq_min)
        log_max = np.log10(GW_freq_max)
        log_freq = log_min + utheta[4] * (log_max - log_min)
        transformed_params.append(10**log_freq)

        
        # Transform costheta (uniform from -1 to 1)
        costheta_min, costheta_max = prior_ranges['costheta']
        transformed_params.append(costheta_min + utheta[5] * (costheta_max - costheta_min))

        #costheta = costheta_min + utheta[5] * (costheta_max - costheta_min)
        #theta = np.arccos(costheta)  # theta will be in the range [0, π]
        #transformed_params.append(theta)

        # Transform phi (uniform from 0 to 2π)
        phi_min, phi_max = prior_ranges['phi']
        transformed_params.append(phi_min + utheta[6] * (phi_max - phi_min))

        # rms_factor ← NEW PARAMETER
        ##rms_min, rms_max = prior_ranges['rms_factor']
        ##transformed_params.append(rms_min + utheta[7] * (rms_max - rms_min))

        # rms_factor ← sample in log-space
        #rms_min, rms_max = prior_ranges['rms_factor']  # e.g. [0.1, 10]
        #log_rms_min = np.log10(rms_min)
        #log_rms_max = np.log10(rms_max)
        #log_rms = log_rms_min + utheta[7] * (log_rms_max - log_rms_min)
        #transformed_params.append(10 ** log_rms)
        transformed_params.append(1.0)

        # Now append per-pulsar red noise parameters.
        # We assume the next entries in utheta are:
        # [u_log10A_p0, u_gamma_p0, u_log10A_p1, u_gamma_p1, ...]
        idx = 8  # next index after the 7 base params; change if different
        #Np = self._n_pulsars  # or set Np = config['n_pulsars']
        #Np = 20  # or set Np = config['n_pulsars']
        Np = PTA_sim._n_pulsars
        for p in range(Np):
            log10_min, log10_max = prior_ranges[f'logA_red_p{p}']
            u_log10 = utheta[idx]; idx += 1
            log10A = log10_min + u_log10 * (log10_max - log10_min)
            A_red = 10.0**log10A
            transformed_params.append(A_red)

            gamma_min, gamma_max = prior_ranges[f'gamma_red_p{p}']
            u_gamma = utheta[idx]; idx += 1
            gamma = gamma_min + u_gamma * (gamma_max - gamma_min)
            transformed_params.append(gamma)
            
        
        return np.array(transformed_params)


    # **Changed**: Initialize Dynesty sampler
    ndim = len(config['prior_or_value'])  # Adjust if necessary
    print(f'ndim: {ndim}')
    sampler = NestedSampler(loglikelihood=mod.ll_func,
                            prior_transform=prior_transform,
                            ndim=ndim,
                            #nlive=sampler_opts['nlive'],
                            nlive=4000,
                            bound='multi',
                            walks=100,
                            sample='rwalk')

    print(f"Running Dynesty with output directory {outdir}")

    # **Changed**: Run the Dynesty sampler
    #sampler.run_nested(dlogz=0.5, maxcall=15_000_000, maxiter=150_000)
    sampler.run_nested(dlogz=0.05)

    # **Changed**: Get results from Dynesty
    results = sampler.results

    weights = np.exp(results.logwt - results.logz[-1])
    
    # **Changed**: Save posterior samples (modify path or file format as needed)
    #posterior_samples = results.samples
    # Weighted resampling to get unweighted posterior samples
    posterior_samples = dyutils.resample_equal(results.samples, weights)

    logweights = results.logwt  # Log of the weights for each sample
    
    # **Changed**: Save the samples
    np.savetxt(os.path.join(outdir, 'posterior_samples.txt'), posterior_samples)
    np.savetxt(os.path.join(outdir, 'logweights.txt'), logweights)
    np.savetxt(os.path.join(outdir, 'weights.txt'), weights)
