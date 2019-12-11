""""Contains calculations of data analysis of the time to catastrophe assuming it is gamma distributed
    Methods included:
    
        log_like_gamma:
        Implementation of the log-likelihood function of the gamma distribution.
        - arguments:
            params: type(tuple); paramters alpha and beta of the gamma distribution (alpha = number of processes, beta = rate of process)
            t: data set of time points to calculate the gamma distribution from
    
        mle_gamma:
        Calculates the maximum likelihood estimate of the paramters alpha and beta of the gamma distribution given the data t
        - arguments:
            t: data set to calculate the maximum likelihood estimate from
            guess: type(list) & len(list = 2); initial guess for the MLE of alpha and beta
        
        log_like_gamma_fixed:
        Implementation of the log-likelihood function of the gamma distribution if the number of poisson processes is known.
        - arguments:
            b: parameter beta of the gamma distribution (rate)
            t: data set of time points to calculate the gamma distribution from
            a: parameter alpha of the gamma distribution (default number of processes is set to 2)

        mle_gamma_fixed:
        Calculates the maximum likelihood estimate of the paramter beta of the gamma distribution given the data t and a fixed number of processes alpha
        - arguments:
            t: data set to calculate the maximum likelihood estimate from
            guess: initial guess for the MLE of beta
            
        draw_bs_sample_gamma:
        Draw a bootstrap sample from a 1D data set.
        - arguments: 
            data: data to draw bootstrap replicates from
            
       draw_bs_reps_mle_gamma:
       Draw nonparametric bootstrap replicates of maximum likelihood estimator.
       - arguments:
            mle_fun : Function with call signature mle_fun(data, *args) that computes a MLE for the parameters
            data : one-dimemsional Numpy array; Array of measurements
            args : tuple, default (); Arguments to be passed to `mle_fun()`.
            size : int, default 1; Number of bootstrap replicates to draw.
            progress_bar : bool, default = False; Whether or not to display progress bar.
        
        get_mle_confidence_gamma:
        Returns the confidence interval for the input mle
        - arguments:
            times: array of times measured for time to catastrophe
            
        """

#Preamb 

#Data analysis
import warnings

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats as st
import scipy.special as sp

#Tools
from tqdm import tqdm


# calculates the log likelihood for alpha and beta in the gamma distribution
def log_like_gamma(params, t):
    alpha, b = params

    if alpha <= 0 or b < 0:
        return -np.inf

    return np.sum(st.gamma.logpdf(t, alpha, scale=1 / b))


# calculates the MLE for alpha and beta
def mle_gamma(t, guess = [2, 0.4]):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, t: -log_like_gamma(params, t),
            x0=np.array(guess),
            args=(t,),
            method="Powell",
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError("Convergence failed with message", res.message)
        
        
# this function calculates the log likelihood when alpha is fixed
def log_like_gamma_fixed(b, t, a = 2):
    alpha = a

    if b < 0:
        return -np.inf

    return np.sum(st.gamma.logpdf(t, alpha, scale=1 / b))

# calculates the MLE when alpha is fixed
def mle_gamma_fixed(t, guess):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda b, t: -log_like_gamma_fixed(b, t),
            x0=np.array([guess]),
            args=(t,),
            method="Powell",
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError("Convergence failed with message", res.message)
        
        
# boostrap functions from BE/Bi103a lesson 8
rg = np.random.default_rng()


def draw_bs_sample_gamma(data):
    """Draw a bootstrap sample from a 1D data set."""
    return rg.choice(data, size=len(data))


def draw_bs_reps_mle_gamma(mle_fun, data, args=(), size=1, progress_bar=False):
    """Draw nonparametric bootstrap replicates of maximum likelihood estimator.

    Parameters
    ----------
    mle_fun : function
        Function with call signature mle_fun(data, *args) that computes
        a MLE for the parameters
    data : one-dimemsional Numpy array
        Array of measurements
    args : tuple, default ()
        Arguments to be passed to `mle_fun()`.
    size : int, default 1
        Number of bootstrap replicates to draw.
    progress_bar : bool, default False
        Whether or not to display progress bar.

    Returns
    -------
    output : numpy array
        Bootstrap replicates of MLEs.
    """
    if progress_bar:
        iterator = tqdm(range(size))
    else:
        iterator = range(size)

    return np.array([mle_fun(draw_bs_sample_gamma(data), *args) for _ in iterator])

def get_mle_confidence_gamma(times, confidence = 0.95, alpha_fixed = True):
    
    bound = (1 - confidence)/2
    upper_b = confidence + bound
    lower_b = 1 - upper_b
    if alpha_fixed:
        # drawing 5000 boostrap replicates for alpha and beta
        bs_reps = draw_bs_reps_mle_gamma(mle_gamma_fixed, times, size=5000, progress_bar=True)

        #Get confidence interval
        conf_int = np.percentile(bs_reps, [lower_b, upper_b], axis=0)
    
    else:
        # drawing 5000 boostrap replicates for alpha and beta
        bs_reps = draw_bs_reps_mle_gamma(mle_gamma, times, size=5000, progress_bar=True)

        #Get confidence interval
        conf_int = np.percentile(bs_reps, [lower_b, upper_b], axis=0)
        
    return conf_int
