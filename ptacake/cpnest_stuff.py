"""
Created on Mon Aug  5 12:49:59 2019

@author: jgoldstein
"""
import numpy as np
import os

import cpnest
import cpnest.model


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
        
    def __init__(self, prior_or_value, PTA_sim, ll_name, 
                 model_func=None, model_names=[], model_kwargs={}):
        """
        prior_or_value: dict
            keys must be all parameter names in model_names, plus 'theta', 'phi'
            for the source location. Each value must either be a list or tuple of 
            [lower, upper] bounds for the parameter if it is to be sampled,
            or a "true" value if it is to be fixed.
        PTA_sim: PTA_sim object
        ll_name: str 
            one of {'TD', 'TD_ns', 'FD', 'FD_ns'}
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
                    'FD_ns':PTA_sim.log_likelihood_FD_ns,
                    'FD_null':PTA_sim.log_likelihood_FD_onlynull
                    }
        
        # things we need in the log_likelihood function
        self.ll_name = ll_name
        self.ll_func = ll_funcs[ll_name]
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
                
            # first check the parameter name (without "log") is a valid model parameter name
            if not pname in self.model_names + ['theta', 'phi']:
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
                
        # check we have an entry for each model parameter and theta, phi
        for p in self.model_names + ['theta', 'phi']:
            if not (p in self.current_values or ('log' + p) in self.current_values):
                raise ValueError('Missing fixed value or prior for parameter {} in prior dict'.format(p))
            if not p in self.current_values_nolog:
                raise ValueError('Missing fixed value or prior for parameter {} in prior dict'.format(p))
                
        # things that cpnest uses
        self.names = sample_names
        self.bounds = sample_bounds
        

    def log_likelihood(self, livepoint):
        """
        Evaluate the likelihood on a livepoint
        """
        self.current_values.update(livepoint)
        
        # go through params and convert log into not log
        for key, value in self.current_values.items():
            if key[:3] == 'log':
                self.current_values_nolog.update({key[3:]:10**value})
            else:
                self.current_values_nolog.update({key:value})
            
        # split off theta and phi parameters from other model parameters
        source = [self.current_values['theta'], self.current_values['phi']]    
        
        # for FD_null likelihood, don't have model and model_args etc
        if self.ll_name == 'FD_null':
            ll = self.ll_func(source)
        
        else:
            # get the rest of the model parameters in the correct order
            # then pass to likelihood function
            model_args = [self.current_values_nolog[p] for p in self.model_names]
            ll = self.ll_func(source, self.model_func, model_args, add_norm=True, 
                              return_only_norm=False, **self.model_kwargs)

        return ll
    


def run(PTA_sim, config, outdir='./output'):
    
    # for FD_null likelihood, don't need GW model etc
    if config['ll_name'] == 'FD_null':
        mod = cpnest_model(config['prior_or_value'], PTA_sim, config['ll_name'])
    
    else:
        if config['model_name'] in ['sinusoid', 'Sinusoid', 'sinusoid_TD', 'Sinusoid_TD']:
            from .GW_models import sinusoid_TD
            model_func = sinusoid_TD
            model_names = ['phase', 'amp', 'pol', 'cosi', 'GW_freq']
        else:
            raise NotImplementedError('Other models than GW sinusoid currently not implemented')
        
        mod = cpnest_model(config['prior_or_value'], PTA_sim, config['ll_name'], 
                          model_func=model_func, model_names=model_names)
    
    sampler_opts = config['sampler_opts']
    
    # check if environment variable SLURM_NTASKS exists. If so, use that for nthreads
    if 'SLURM_NTASKS' in os.environ:
        nthreads = int(os.environ['SLURM_NTASKS'])
    # otherwise, use nthreads from config
    else:
        nthreads = sampler_opts['nthreads']
    
    #Instantiate the sampler
    cpn = cpnest.CPNest(usermodel=mod,
                        nlive=sampler_opts['nlive'],
                        maxmcmc=sampler_opts['nsteps'],
                        nthreads=nthreads,
                        verbose=3,
                        output=outdir,
                        resume=sampler_opts['resume'])
    
    print('Putting cpnest output in {}'.format(cpn.output))
    print('Running CPNest!\n')
    cpn.run()
    
    # save the posterior samples
    cpn.get_nested_samples()
    cpn.get_posterior_samples()
    