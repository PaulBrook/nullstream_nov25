"""
Created on Tue Jun 18 14:50:47 2019

@author: jgoldstein

likelihoods for Time Domain (with and without null-streaming), and Frequency Domain
without null-streaming. FD + ns is in the _PTA_sim_nullstream module because it 
requires some more steps (concatenating).
"""
import numpy as np
import sys

#from .nullstream_algebra import null_streams, response_matrix
from ptacake.nullstream_algebra import null_streams, response_matrix

l2p = np.log(2*np.pi)
hl2p = 0.5 * np.log(2*np.pi)

    
def log_likelihood_TD_es(self, source, model_func, model_args, 
                         add_norm=False, return_only_norm=False, **model_kwargs):
    """
    Time domain log likelihood, only for evenly sampled data.
    """
    
    # assuming evenly sampled times, same for all pulsars
    times = self._times[0]
    model_hplus, model_hcross = model_func(times, *model_args, **model_kwargs)
    responses = response_matrix(*source, self._pulsars[['theta', 'phi']].values)
    Fplus = np.expand_dims(responses[:, 0], -1)
    Fcross = np.expand_dims(responses[:, 1], -1)
    model = Fplus * model_hplus + Fcross * model_hcross
    
    # take inner product of residuals - model with itself
    x = self.residuals - model
    inv_cov = np.diag(self._pulsars['rms'].values**(-2))
    product = np.einsum('i...,ik,k...', x, inv_cov, x)
    ll = -0.5 * np.sum(product) # sum over times
    
    
    if add_norm or return_only_norm:
        # use log(det(cov)) = -log(det(inv_cov))
        sign, logdet = np.linalg.slogdet(inv_cov)
        norm = len(times) * (-self._n_pulsars*hl2p + 0.5*logdet)
        if return_only_norm:
            return norm
        ll += norm
        
    return ll

def compute_snr(self):
    """
    Compute signal to noise ratio of injected signal, assuming
    noise level specified by pulsar rms values. Computation in the 
    time domain.
    
    Returns
    -------
    float:
        Signal to noise ratio of injected signal
    """
    snr2 = 0
    for p in range(self._n_pulsars):
        # get mask for which times are not nan, and select non-nan signal points
        mask = np.isfinite(self._times[p])
        signal = self._signal[p][mask]
        
        # compute inner product of signal with itself, wrt noise covariance
        inv_cov = self._TD_inv_covs[p]
        product = np.einsum('i,ij,j', signal, inv_cov, signal)
        snr2 += product
        
    return np.sqrt(snr2)
    

def log_likelihood_TD(self, source, model_func, model_args, 
                      add_norm=False, return_only_norm=False, **model_kwargs):
    """
    Time domain log likelihood that works for any sampling of data.
    """
    # Because the number of times (that aren't nan) for each pulsar can be different,
    # we will need to do some part of the computation per perulsar (for-loop below)
    # but we can precompute the pulsar responses
    responses = response_matrix(*source, self._pulsars[['theta', 'phi']].values)
    Fplus = responses[:, 0]
    Fcross = responses[:, 1]
    # and we can also precompute the model, using the times that include nans
    model_hplus, model_hcross = model_func(self._times, *model_args, **model_kwargs)
    
    # compute logl and norm by summing contributions per pulsar
    ll = 0
    norm = 0
    
    for p in range(self._n_pulsars):
        # get mask for which times are not nan
        mask = np.isfinite(self._times[p])
        num_times = np.sum(mask)
        # select model for this pulsar, and use mask to select points, apply responses
        model = Fplus[p] * model_hplus[p][mask] + Fcross[p] * model_hcross[p][mask]
        
        # compute product of (res - model) * inv_covariance * (res - model)
        x = self.residuals[p][mask] - model
        inv_cov = self._TD_inv_covs[p]
        product = np.einsum('i,ij,j', x, inv_cov, x)
        ll += -0.5 * np.sum(product) # np.sum sums over times
    
        if add_norm or return_only_norm:    
            # use log(det(cov)) = -log(det(inv_cov))
            sign, logdet_inv_cov = np.linalg.slogdet(inv_cov)
            norm += -num_times*hl2p + 0.5*logdet_inv_cov
    
    if return_only_norm:
        return norm
    
    if add_norm:
        ll += norm
        
    return ll
    

def log_likelihood_TD_ns(self, source, model_func, model_args, 
                         add_norm=False, return_only_norm=False, **model_kwargs):
    """
    Time domain null-stream likelihood only possible for evenly sampled times.
    """
    
    # convert residuals data to null streams
    # transform inverse covariance as well
    pulsar_array = self._pulsars[['theta', 'phi']].values
    inv_cov = np.diag(self._pulsars['rms'].values**(-2))
    ns_data, ns_inv_cov = null_streams(self.residuals, inv_cov, 
                                       source, pulsar_array)
    
    # assuming evenly sampled times, same for all pulsars
    times = self._times[0]
    model_hplus, model_hcross = model_func(times, *model_args, **model_kwargs)
    
    # combine hplus, hcross with null streams to get full model
    nulls = (np.zeros_like(model_hplus),)*(self._n_pulsars-2)
    ns_model = np.hstack((model_hplus, model_hcross, *nulls)).reshape(
                                       self._n_pulsars, len(model_hplus))

    # take product of "null-streamed" residuals - model
    x = ns_data - ns_model
    product = np.einsum('i...,ik,k...', x, ns_inv_cov, x)
    ll = -0.5 * np.sum(product) # sum over times
    
    
    if add_norm or return_only_norm:
        # for normalisation, use the inv_cov WITHOUT null-stream transformation
        # use log(det(cov)) = -log(det(inv_cov))
        sign, logdet = np.linalg.slogdet(inv_cov)
        norm = len(times) * (-self._n_pulsars*hl2p + 0.5*logdet)
        if return_only_norm:
            return norm
        ll += norm
    
    return ll

   
def log_likelihood_FD(self, source, model_func, model_args, 
                      add_norm=False, return_only_norm=False, **model_kwargs):

    #model_args = [6.210865757043734448e+00, 7.110201957367899311e-13, 5.216693746564841083e-01, 7.017131688962230385e-03, 2.205540718312692965e-10]
    #source = [np.arccos(1.046660166710000617e-02), 1.655887861229982327e+00]

    ###print(f'Params (model_args): {model_args}')
    ###print(f'Source: {source}')
    
    # call model function with args and preset model times, then funky fourier
    fourier_hplus, fourier_hcross = self.fourier_model(model_func, 
                                                *model_args, **model_kwargs)
    
    # apply response functions
    responses = response_matrix(*source, self._pulsars[['theta', 'phi']].values)
    Fplus = np.expand_dims(responses[:, 0], -1)
    Fcross = np.expand_dims(responses[:, 1], -1)
    # Npulsars x Nfreqs
    model = Fplus * fourier_hplus + Fcross * fourier_hcross
    # compute logl and norm by summing contributions per pulsar
    ll = 0
    norm = 0
    
    for p in range(self._n_pulsars):
        #print(f"Processing pulsar {p+1}/{self._n_pulsars}")
        # compute product of (res - model) * inv_covariance * (res - model)
        #print(f'RES:{self.residualsFD[p]}')
        #print(f'MOD:{model[p]}')
        x = self.residualsFD[p] - model[p]
        inv_cov = self._TOA_FD_inv_covs[p]
        product = np.einsum('a,ab,b', x, inv_cov, np.conj(x))
        #print(f"Product: {product}")
        # we think there should be a factor of 2 here to compensate for missing negative frequencies
        ll += -0.5 * 2 * np.real(np.sum(product)) # np.sum sums over frequencies

        if add_norm or return_only_norm:
            # use log(det(cov)) = -log(det(inv_cov))
            sign, logdet = np.linalg.slogdet(inv_cov)
            # no 1/2 in norm because we have norm for real and for imag part
            norm += -self._n_freqs*l2p + logdet
        
    if return_only_norm:
        return norm        
    
    if add_norm:
        ll += norm

    ###print(f"Final log-likelihood: {ll}")    
    return ll


def red_psd_from_A_gamma(A, gamma, freqs, f_ref):
    """
    Compute red-noise PSD for timing residuals in units of s^2/Hz.

    Parameters
    ----------
    A : float
        Amplitude of red noise (dimensionless, typically ~1e-13).
    gamma : float
        Spectral index (usually ~4).
    freqs : array
        Frequency array (Hz).
    f_ref : float
        Reference frequency, usually 1/year in Hz.

    Returns
    -------
    psd : array
        Power spectral density values at each frequency (s^2/Hz).
    """
    return (A**2 / (12.0 * np.pi**2)) * (freqs / f_ref)**(-gamma) * (freqs**(-3))


def log_likelihood_FD_dyn(self, params, add_norm=False, return_only_norm=False, **model_kwargs):

    from .GW_models import sinusoid_TD
    model_func = sinusoid_TD # hardcoded here to get things working

    #params = [6.210865757043734448e+00, 7.110201957367899311e-13, 5.216693746564841083e-01, 7.017131688962230385e-03, 2.205540718312692965e-10, 1.046660166710000617e-02, 1.655887861229982327e+00]
    
    ###print(f"Parameters: {params}")
    rms_factor = params[7]

    # Remaining params are per-pulsar red noise params
    ###print(f'PARAMS: {params}')
    red_params = params[8:]
    ###print(f'RED PARAMS: {red_params}')
    red_params = np.array(red_params).reshape(self._n_pulsars, 2)

    print(f'Red parameters: {red_params}')
    
    # call model function with args and preset model times, then funky fourier
    fourier_hplus, fourier_hcross = self.fourier_model(model_func,
                                                *params[:6], **model_kwargs)

    #print(f"Fourier hplus shape: {fourier_hplus.shape}, Fourier hcross shape: {fourier_hcross.shape}")
    
    if params[5] == 0:
        source = [params[5],params[6]]
    else:
        source = [np.arccos(params[5]),params[6]]
    ###print(f"Source: {source}")
    
    # apply response functions
    responses = response_matrix(*source, self._pulsars[['theta', 'phi']].values)
    Fplus = np.expand_dims(responses[:, 0], -1)
    Fcross = np.expand_dims(responses[:, 1], -1)
    #print(f"Fplus shape: {Fplus.shape}, Fcross shape: {Fcross.shape}")
    # Npulsars x Nfreqs
    model = Fplus * fourier_hplus + Fcross * fourier_hcross
    #print(f"Model shape: {model.shape}")

    # compute logl and norm by summing contributions per pulsar
    ll = 0
    norm = 0

    for p in range(self._n_pulsars):
        #### compute product of (res - model) * inv_covariance * (res - model)
        ###x = self.residualsFD[p] - model[p]
        ####inv_cov = self._TOA_FD_inv_covs[p]
        ###inv_cov = self._TOA_FD_inv_covs[p] / (rms_factor**2)
        ###product = np.einsum('a,ab,b', x, inv_cov, np.conj(x))
        #### we think there should be a factor of 2 here to compensate for missing negative frequencies
        ###ll += -0.5 * 2 * np.real(np.sum(product)) # np.sum sums over frequencies
        #### Always validate the determinant of the inverse covariance matrix
        ####sign, logdet = np.linalg.slogdet(inv_cov)

        #log10_A, gamma = red_params[p]
        A_red, gamma = red_params[p]
        print(f'For pulsar {p}, A_red is {A_red} and gamma is {gamma}')
        #A_red = 10.0**log10_A
        A_red = float(A_red)

        # Build red noise PSD for this pulsar
        red_psd = red_psd_from_A_gamma(A_red, gamma,
                                       self._freqs,
                                       f_ref=1.0 / (365.25 * 24 * 3600))
        
        # Debug: print some stats about red_psd
        ###print(f"[Pulsar {p}] A_red={A_red:.3e}, gamma={gamma:.3f}, "
        ###      f"red_psd min={red_psd.min():.3e}, max={red_psd.max():.3e}")

        
        cov = self._TOA_FD_covs[p] + np.diag(red_psd)   # total covariance

        # Debug: eigenvalues of covariance
        eigvals = np.linalg.eigvalsh(cov)
        ###print(f"[Pulsar {p}] cov eig min={eigvals.min():.3e}, "
        ###      f"max={eigvals.max():.3e}, cond={eigvals.max()/eigvals.min():.3e}")

        print(f"Pulsar {p}:")
        print(f"  red_psd range = {red_psd.min():.3e} – {red_psd.max():.3e}")
        print(f"  white cov diag mean = {np.mean(np.diag(self._TOA_FD_covs[p])):.3e}")
        print(f"  total cov diag mean = {np.mean(np.diag(cov)):.3e}")
        
        inv_cov = np.linalg.inv(cov) / (rms_factor**2)

        # Residuals minus model for this pulsar
        x = self.residualsFD[p] - model[p]

        # Quadratic form
        product = np.einsum('a,ab,b', x, inv_cov, np.conj(x))

        # Factor of 2 for missing negative frequencies
        ll += -0.5 * 2 * np.real(np.sum(product))
        
        if add_norm or return_only_norm:
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                raise ValueError("Covariance not positive definite for pulsar {}".format(p))
            # Include scaling by rms_factor
            logdet += self._n_freqs * np.log(rms_factor**2)
            norm += -self._n_freqs * l2p + logdet

            
            #### use log(det(cov)) = -log(det(inv_cov))
            #### Adjust log(det(cov)) for scaling by rms_factor^2
            ###logdet = self._TOA_FD_cov_logdets[p] + self._n_freqs * np.log(rms_factor**2)
            ####sign, logdet = np.linalg.slogdet(inv_cov)
            #### no 1/2 in norm because we have norm for real and for imag part
            ###norm += -self._n_freqs*l2p + logdet
            ####print(f"Norm after pulsar {p+1}: {norm}")

    if return_only_norm:
        return norm

    if add_norm:
        ll += norm
        #print(f"ll with norm added: {ll}")

    print(f"Final log-likelihood: {ll}")
    return ll

functions = [log_likelihood_TD, log_likelihood_TD_ns, log_likelihood_TD_es,
             log_likelihood_FD, log_likelihood_FD_dyn, compute_snr]
