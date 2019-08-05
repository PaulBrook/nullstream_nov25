"""
Created on Mon Aug  5 12:49:59 2019

@author: jgoldstein
"""
import numpy as np
import os

import cpnest
import cpnest.model

import ptacake as cake


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
                 model_func, model_names, model_kwargs):
        """
        prior_or_value: dict
            keys must be all parameter names in model_names, plus 'theta', 'phi'
            for the source location. Each value must either be a list or tuple of 
            [lower, upper] bounds for the parameter if it is to be sampled,
            or a "true" value if it is to be fixed.
        PTA_sim: PTA_sim object
        ll_name: str ['TD', 'TD_ns', 'FD', 'FD_ns']
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
                    'FD_ns':PTA_sim.log_likelihood_FD_ns
                    }
        
        # things we need in the log_likelihood function
        self.ll_func = ll_funcs[ll_name]
        self.model_func = model_func
        self.model_names = model_names
        self.model_kwargs = model_kwargs
        
        # go through the "priors_or_values" dict to see which params are sampled
        sample_names = []
        sample_bounds = []
        # the dict "current_values" will hold the "true" values for now, then
        # in the likelihood evaluation we will update it with livepoint
        self.current_values = {}
        for key, value in prior_or_value.items():
            # check if iterable with lower, upper is given
            try:
                lower, upper = value
                sample_names.append(key)
                sample_bounds.append([lower, upper])
                self.current_values.update({key:None})
            except:
                self.current_values.update({key:value})
                
        # things that cpnest uses
        self.names = sample_names
        self.bounds = sample_bounds
        

    def log_likelihood(self, livepoint):
        """
        Evaluate the likelihood on a livepoint
        """
        self.current_values.update(livepoint)
        
        source = [self.current_values['theta'], self.current_values['phi']]
        model_args = [self.current_values[p] for p in self.model_names]
        ll = self.ll_func(source, self.model_func, model_args, add_norm=True, 
                          return_only_norm=False, **self.model_kwargs)

        return ll
    


def run(PTA_sim, config):
    
    if config['model_name'] in ['sinusoid', 'Sinusoid', 'sinusoid_TD', 'Sinusoid_TD']:
        from ptacake.GW_models import sinusoid_TD
        model_func = sinusoid_TD
        model_names = ['phase', 'amp', 'pol', 'cosi', 'GW_freq']
        model_kwargs = {}
    else:
        raise NotImplementedError('Other models than GW sinusoid currently not implemented')
    
    mod = cpnest_model(config['prior_or_value'], PTA_sim, config['ll_name'], 
                      model_func, model_names, model_kwargs)
    
    file_dir = os.path.dirname(config['outfile_path'])
    
    #Instantiate the sampler
    sampler_opts = config['sampler_opts']
    cpn = cpnest.CPNest(usermodel=mod,
                        nlive=sampler_opts['nlive'],
                        maxmcmc=sampler_opts['nsteps'],
                        nthreads=sampler_opts['nthreads'],
                        verbose=3,
                        output=file_dir,
                        resume=sampler_opts['resume'])
    
    print('Running CPNest!\n')
    cpn.run()
    
    # save the posterior samples
    cpn.get_nested_samples()
    cpn.get_posterior_samples()
    