"""
Created on Wed Aug 14 13:32:55 2019

@author: jgoldstein

grid sampler
"""
import os
import numpy as np

class Grid_sampler():
    def __init__(self, prior_or_value, PTA_sim, ll_name, 
                 model_func, model_names, model_kwargs):
        
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
        
        
    def loglikelihood(self, point):
        """
        Evaluate the likelihood on a point (dict)
        """
        self.current_values.update(point)
        
        # go through params and convert log into not log
        for key, value in self.current_values.items():
            if key[:3] == 'log':
                self.current_values_nolog.update({key[3:]:10**value})
            else:
                self.current_values_nolog.update({key:value})
                
        # split values into source (theta, phi) and a list of model params 
        # in the correct order, to pass onto the likelihood function
        source = [self.current_values['theta'], self.current_values['phi']]
        model_args = [self.current_values_nolog[p] for p in self.model_names]
        ll = self.ll_func(source, self.model_func, model_args, add_norm=True, 
                          return_only_norm=False, **self.model_kwargs)

        return ll
        
    
    def run(self, ngrid=10, output='grid_out', save_interm=False):
        """
        Sample log likelihood over a grid.
        """
        ranges = {}
        for name, bound in zip(self.names, self.bounds):
            r = np.linspace(bound[0], bound[1], num=ngrid)
            print('param {} with range {}'.format(name, r))
            ranges[name] = r
            
        with open(os.path.join(output, 'grid_interm.txt'), 'a') as interm:
            param_cols = ('{}\t'*len(self.names)).format(*self.names)
            interm.write('idx\t'+param_cols+'log_like\n')
        
            log_likes = np.zeros((ngrid,)*len(self.names))
            for idx in np.ndindex(log_likes.shape):
                
                point = {}
                for pi, vi in enumerate(idx):
                    pname = self.names[pi]
                    point[pname] = ranges[pname][vi]
                    
                log_likes[idx] = self.loglikelihood(point)
                
                if save_interm:
                    interm.write('{}\t'.format(idx))
                    for pname in self.names:
                        interm.write('{}\t'.format(point[pname]))
                    interm.write('{}\n'.format(log_likes[idx]))
        
        np.savez(os.path.join(output, 'grid_log_likelihoods.npz'), 
                 log_likelihoods=log_likes, **ranges)


def run(PTA_sim, config):
    
    if config['model_name'] in ['sinusoid', 'Sinusoid', 'sinusoid_TD', 'Sinusoid_TD']:
        from .GW_models import sinusoid_TD
        model_func = sinusoid_TD
        model_names = ['phase', 'amp', 'pol', 'cosi', 'GW_freq']
        model_kwargs = {}
    else:
        raise NotImplementedError('Other models than GW sinusoid currently not implemented')
        
    grid_sampler = Grid_sampler(config['prior_or_value'], PTA_sim, config['ll_name'], 
                      model_func, model_names, model_kwargs)
    
    file_dir = config['output_path']
    assert(os.path.exists(file_dir))
    sampler_opts = config['sampler_opts']
    
    grid_sampler.run(ngrid=sampler_opts['nsteps'], output=file_dir, save_interm=sampler_opts['resume'])