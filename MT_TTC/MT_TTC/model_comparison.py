"""Contains functions necessary to compare the gamma model and the model encompassing two poisson processes with different rates
    Methods included:
        
        log_like_gamma_2(params, t):
            Implementation of the log-likelihood function of the gamma distribution.
            - arguments:
            params: type(tuple); paramters alpha and beta of the gamma distribution (alpha = number of processes, beta = rate of process)
            t: data set of time points to calculate the gamma distribution from

        log_like_ttc_distribution_2(params, t):
            Implementation of the log-likelihood function of the TTC distribution.
            - arguments:
            params: type(tuple); paramters beta_1 and beta_2 of the gamma distribution (rates of process 1 and 2)
            t: data set of time points to calculate the TTC distribution from

        mle_gamma2(t):
            Calculates the maximum likelihood estimate of the paramters alpha and beta of the gamma distribution given the data t
            - arguments:
            t: data set to calculate the maximum likelihood estimate from
            guess: type(list) & len(list = 2); initial guess for the MLE of alpha and beta

        draw_bs_sample2(data):
           Draw a bootstrap sample from a 1D data set.
            - arguments: 
            data: data to draw bootstrap replicates from

        def draw_bs_reps_mle2(mle_fun, data, args=(), size=1, progress_bar=False):
            Draw nonparametric bootstrap replicates of maximum likelihood estimator.
            - arguments:
            mle_fun : Function with call signature mle_fun(data, *args) that computes a MLE for the parameters
            data : one-dimemsional Numpy array; Array of measurements
            args : tuple, default (); Arguments to be passed to `mle_fun()`.
            size : int, default 1; Number of bootstrap replicates to draw.
            progress_bar : bool, default = False; Whether or not to display progress bar.

        AIC_Akaike(params_gamma, params_ttc, times):
            Calculates AIC for both distributions and Akaike weight for the gamma distribution.
            -arguments:
            params_gamma : alpha and beta MLE
            params_TTC : beta1 and beta2 MLE
            times : one-dimemsional Numpy array; Array of measurements

        def gamma_params_noprint(data):
            Calculates all gamma parameters and confidence intervals
            -arguments:
            data : one-dimemsional Numpy array; Array of measurements

        def draw_gamma(alpha, b, size=1):
            draws bootstrap samples of size
            -arugmnents:
                alpha: alpha of gamma distribution
                beta: beta of gamma distribution
                size: how many points 

        def ecdf2(x, data):
           calculates the ecdf of given data
           -arguments:
           x : arbitrary points
           data : one-dimemsional Numpy array; Array of measurements

        def predictive_ecdf_gamma(params, data, plot_data, data_size):
            bootstrap to calculate predictive ecdf for gammma distribution
            -arguments:
            params : alpha and beta MLE
            data : one-dimemsional Numpy array; Array of measurements
            plot_data : dataframe storing times
            data_size : how many points

        def draw_ttc(beta_1, beta_2, size=1):
            draws bootstrap samples of size
            -arugmnents:
                beta1: beta1 of ttc distribution
                beta2: beta of ttc distribution
                size: how many points 

        def predictive_ecdf_ttc(params, data, plot_data, data_size):
            bootstrap to calculate predictive ecdf for ttc distribution
            -arguments:
            params : beta1 and beta2 MLE
            data : one-dimemsional Numpy array; Array of measurements
            plot_data : dataframe storing times
            data_size : how many points
"""

#Preamb

#Data analysis
import warnings
import numba
import math

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats as st
import scipy.special as sp

#Plotting
import bokeh.io
import bokeh.plotting
import bokeh_catplot
import holoviews as hv
import bebi103

bokeh.io.output_notebook()

hv.extension('bokeh')

#Tools
import black
from tqdm import tqdm

# calculates the log likelihood for alpha and beta in the gamma distribution
def log_like_gamma_2(params, t):
    alpha, b = params

    if alpha <= 0 or b < 0:
        return -np.inf

    return np.sum(st.gamma.logpdf(t, alpha, scale=1 / b))

# calculates the log likelihood for beta_1 and beta_2 in the given distribution
def log_like_ttc_distribution_2(params, t):
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

#calculates the MLE for alpha and beta
def mle_gamma2(t):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, t: -log_like_gamma_2(params, t),
            x0=np.array([2, 0.4]),
            args=(t,),
            method='Powell'
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)

#boostrap functions from BE/Bi103a lesson 8
rg = np.random.default_rng()

def draw_bs_sample2(data):
    """Draw a bootstrap sample from a 1D data set."""
    return rg.choice(data, size=len(data))

def draw_bs_reps_mle2(mle_fun, data, args=(), size=1, progress_bar=False):
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

    return np.array([mle_fun(draw_bs_sample2(data), *args) for _ in iterator])

def AIC_Akaike(params_gamma, params_ttc, times):
    #calculate log likelihood
    log_likelihood_gamma = log_like_gamma_2(params_gamma, times)
    log_likelihood_ttc = log_like_ttc_distribution_2(params_ttc, times)
    
    #calculate AIC
    AIC_gamma = -2*(log_likelihood_gamma-2)
    AIC_ttc = -2*(log_likelihood_ttc-2)

    #calculate akaike weight
    AIC_max = max(AIC_gamma, AIC_ttc)
    numerator = np.exp(-(AIC_gamma - AIC_max)/2)
    denominator = numerator + np.exp(-(AIC_ttc - AIC_max)/2)
    w = numerator / denominator

    return AIC_gamma, AIC_ttc, w


#same wrapper function, but without a print statement. More useful for analyzing multiple datasets. 
def gamma_params_noprint(data):
    #calculate mle parameters
    mle_params = mle_gamma2(data)
    
    #drawing bootstrap reps, 1000 reps for computational time
    bs_reps = draw_bs_reps_mle2(mle_gamma2, data, size=1000, progress_bar=True)
    
    #calculate confidence intervals
    conf_int_gamma = np.percentile(bs_reps, [2.5, 97.5], axis=0)
    
    return mle_params, conf_int_gamma

def draw_gamma(alpha, b, size=1):
    return rg.gamma(alpha, 1/b, size = size)

def ecdf2(x, data):
    """Give the value of an ECDF at arbitrary points x."""
    y = np.arange(len(data) + 1) / len(data)
    return y[np.searchsorted(np.sort(data), x, side="right")]

def predictive_ecdf_gamma(params, data, plot_data, data_size):
    n = len(data)

    #draw data_size datasets with n datapoints from the distribution
    parametric_bs_samples = draw_gamma(params[0], params[1], size = (data_size, n))

    #compute the ECDF value for each value of n
    n_theor = np.arange(0, parametric_bs_samples.max() + 1)
    ecdfs = np.array([ecdf2(n_theor, sample) for sample in parametric_bs_samples])

    #calculate confidence intervals
    ecdf_low, ecdf_high = np.percentile(ecdfs, [2.5, 97.5], axis=0)

    #plot the predictive ecdfs
    p = bebi103.viz.fill_between(
        x1=n_theor,
        y1=ecdf_high,
        x2=n_theor,
        y2=ecdf_low,
        patch_kwargs={"fill_alpha": 0.5},
        x_axis_label="time (s)",
        y_axis_label="ECDF",
        title = 'Predictive ECDF of the gamma distribution'
    )

    #overlay with true data
    p = bokeh_catplot.ecdf(data=plot_data, val='time (s)', palette=['orange'], p=p)

    bokeh.io.show(p)

def draw_ttc(beta_1, beta_2, size=1):
    return rg.exponential(1 / beta_1, size = size) + rg.exponential(1 / beta_2, size = size)

def predictive_ecdf_ttc(params, data, plot_data, data_size):
    n = len(data)

    #draw data_size datasets with n datapoints from the distribution
    parametric_bs_samples = draw_ttc(params[0], params[1], size = (data_size, n))

    #compute the ECDF value for each value of n
    n_theor = np.arange(0, parametric_bs_samples.max() + 1)
    ecdfs = np.array([ecdf2(n_theor, sample) for sample in parametric_bs_samples])

    #calculate confidence intervals
    ecdf_low, ecdf_high = np.percentile(ecdfs, [2.5, 97.5], axis=0)

    #plot the predictive ecdfs
    p = bebi103.viz.fill_between(
        x1=n_theor,
        y1=ecdf_high,
        x2=n_theor,
        y2=ecdf_low,
        patch_kwargs={"fill_alpha": 0.5},
        x_axis_label="time (s)",
        y_axis_label="ECDF",
        title = 'Predictive ECDF of the TTC distribution'
    )

    #overlay with true data
    p = bokeh_catplot.ecdf(data=plot_data, val='time (s)', palette=['orange'], p=p)

    bokeh.io.show(p)

