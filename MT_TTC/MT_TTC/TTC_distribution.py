""""Contains calculations of data analysis of the time to catastrophe assuming it is distributed according to the pdf: p(x) = b_2*b_1)/(b_2-b_1) * (exp(-b_1 x) - exp(-b_2 x))
    Methods included:
    
        log_like_ttc_distribution:
        Implementation of the log-likelihood function of the TTC distribution.
        - arguments:
            params: type(tuple); paramters beta_1 and beta_2 of the gamma distribution (rates of process 1 and 2)
            t: data set of time points to calculate the TTC distribution from
    
        mle_ttc_distribution:
        Calculates the maximum likelihood estimate of the paramters beta_1 and beta_2 of the TTC distribution given the data t
        - arguments:
            t: data set to calculate the maximum likelihood estimate from
            guess: type(list) & len(list = 2); initial guess for the MLE of beat_1 and beta_2
            
        draw_bs_sample_TTC:
        Draw a bootstrap sample from a 1D data set.
        - arguments: 
            data: data to draw bootstrap replicates from
            
       draw_bs_reps_mle_TTC:
       Draw nonparametric bootstrap replicates of maximum likelihood estimator.
       - arguments:
            mle_fun : Function with call signature mle_fun(data, *args) that computes a MLE for the parameters
            data : one-dimemsional Numpy array; Array of measurements
            args : tuple, default (); Arguments to be passed to `mle_fun()`.
            size : int, default 1; Number of bootstrap replicates to draw.
            progress_bar : bool, default = False; Whether or not to display progress bar.
        
        get_mle_confidence_TTC:
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


# calculates the log likelihood for beta_1 and beta_2 in the given distribution
def log_like_ttc_distribution(params, t):
    beta_1, beta_2 = params

    # Condition that rates cannot be negative
    if beta_1 <= 0 or beta_2 <= 0:
        out = -np.inf
        return -np.inf

    # In case beta_1 and beta_2 are close to each other, we use the result from the limit above to approximate both betas as the same.
    elif abs(beta_1 - beta_2) <= 0.005:
        out = np.sum(np.log(t * beta_1 ** 2) - beta_1 * t)
        return out
    # If all conditions are met, we can calculate the log-likelihood
    else:
        out = np.sum(
            np.log((beta_1 * beta_2) / abs(beta_2 - beta_1))
            + np.log(abs(np.exp(-beta_1 * t) - np.exp(-beta_2 * t)))
        )

    return out


# calculates the MLE for beta_1 and beta_2
def mle_ttc_distribution(t, guess = [0.04, 0.06]):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, t: -log_like_ttc_distribution(params, t),
            x0=np.array(guess),
            args=(t,),
            method="Powell",
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError("Convergence failed with message", res.message)
        
        
# boostrap functions from BE/Bi103a lesson 8
rg = np.random.default_rng()
        
    
def draw_bs_sample_TTC(data):
    """Draw a bootstrap sample from a 1D data set."""
    return rg.choice(data, size=len(data))


def draw_bs_reps_mle_TTC(mle_fun, data, args=(), size=1, progress_bar=False):
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

    return np.array([mle_fun(draw_bs_sample_TTC(data), *args) for _ in iterator])


def get_mle_confidence_TTC(times, confidence = 0.95, alpha_fixed = True):
    
    bound = (1 - confidence)/2
    upper_b = confidence + bound
    lower_b = 1 - upper_b
    # drawing 5000 boostrap replicates for alpha and beta
    bs_reps = draw_bs_reps_mle_TTC(mle_TTC, times, size=5000, progress_bar=True)

    #Get confidence interval
    conf_int = np.percentile(bs_reps, [lower_b, upper_b], axis=0)
    
    
    return conf_int
