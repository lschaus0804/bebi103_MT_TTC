""""Contains all plots used to analyze the time to catasrtophe data.
    Methods included:
    
        create_poisson_ecdf:
        Plots the ecdf of a distribution that takes the output from two different poisson distributions and adds them together.
        - arguments:
            beta_1: rate of the first poisson process
            beta_2 rate of the second poisson process
            n_points: number of points to generate
            non_dim: type(bool); non-dimensionalizes the x-axis (default is False)
        
        TTC_cdf_plot:
        Plots the cdf of the distribution p(x) = (b_2*b_1)/(b_2-b_1) * (exp(-b_1 x) - exp(-b_2 x))
        - arguments:
            beta_1: rate of the first poisson process
            beta_2 rate of the second poisson process
            n_points: number of points to generate

        ecdf_vals_plot:
        Takes a pandas data frame of the time to catastrophe data and plots it's ecdf with a confidence interval.
        - arguments:
            df: data frame to be used 
            label: defines the slice of the data
            column_label: defines the column the data is in
            conf_int: type(bool); draws the confidence intervals if set to True (default is True)
            label_bool: type(bool); sets the slicing of the data to True (default is False)
            
        mle_path_plot:
        Uses the bootstraped paramters of a distribution as input and produces a scatter plot along with a confidence contour
        - arguments:
            bs_reps: array of bootstraped parameters
            param_1: type(str); name of the first parameter
            param_2: type(str); name of the second parameter
            conf: confidence interval
            
        mle_confidence_plot:
        Creates a confidence region plot along with the histogram of the parameter distributions of the bootstrap
        -arguments:
            bs_reps: array of bootstraped parameters
            param_1: type(string); name of the first parameter
            param_2: type(string); name of the second parameter
            conf: confidence interval """

#Preamb

#Data Anaylsis
import numpy as np
import pandas as pd
import scipy.stats

#Plotting
from bokeh.plotting import figure, output_file, show

#More plotting
import bokeh.io
import bokeh.plotting
import bokeh_catplot
import holoviews as hv
import bebi103

#Even more plotting
from bokeh.io import show
from bokeh.layouts import row
bokeh.io.output_notebook()
hv.extension('bokeh')

#tools
from tqdm import tqdm
import warnings

#Needed to draw plots for create_poisson_ecdf
def ecdf_vals_viz(
    data,
    beta,
    legend_label="label",
    x_axis_label="x",
    y_axis_label="ECDF",
    non_dim=True,
):

    # Create sorted data and ECDF value
    ecdf_data = np.array([np.sort(data), np.arange(1, len(data) + 1) / len(data)])

    if non_dim == True:
        ecdf_data[0] = ecdf_data[0] * beta

    # Create data frame
    df = pd.DataFrame(np.transpose(ecdf_data), columns=[x_axis_label, y_axis_label])

    # Plot ECDF
    plot = hv.Points(
        data=df,
        kdims=[x_axis_label, y_axis_label],
        # Add label in order to obtain a legend on multiple plots
        label=legend_label,
    )

    return plot


def create_poisson_ecdf(beta_1, beta_2, n_points=150, non_dim=False):

    #Create empty list to append random draws into.
    time_to_catastrophe = []

    #Make random draws from two exponential distributions and add them. Append the result to the empty list. Repeat this for how many points specified.
    for i in range(n_points):
        time_to_catastrophe.append(
            rg.exponential(1 / beta_1) + rg.exponential(1 / beta_2)
        )
    
    #If one wants to non-dimensionalize the ecdf, then non_dim should be set to True, this will also change the x-axis description.
    if non_dim == True:

        ecdf = ecdf_vals_viz(
            time_to_catastrophe,
            beta_1,
            "Approximate CDF",
            "Time to Catastrophe (t\u03B2\N{SUBSCRIPT ONE})",
            non_dim=True,
        )
    else:
        ecdf = ecdf_vals_viz(
            time_to_catastrophe,
            beta_1,
            "Approximate CDF",
            "Time to Catastrophe (t)",
            non_dim=False,
        )

    return ecdf[0].opts(width=400, padding=0.1, xlim=(0, 10))


def true_cdf_plot(beta_1, beta_2, n_points=150000):
    #Generate num points
    x_dimension = np.linspace(0, 250, num=n_points)
    #Generate list of tuples with t and F(t) of the distribution
    cdf = [
        (
            t,
            (beta_1 * beta_2 / (beta_2 - beta_1))
            * (
                (1 / beta_1) * (1 - np.exp(-beta_1 * t))
                - (1 / beta_2) * (1 - np.exp(-beta_2 * t))
            ),
        )
        for t in x_dimension
    ]

    #Plot the above cdf as a scatter plot using holoviews
    true_cdf = hv.Scatter(
        cdf, "Time to catastrophe (t)", "CDF", label="Exact CDF"
    ).opts(
        width=400,
        padding=0.1,
        xlim=(0, 250),
        color=bebi103.hv.default_categorical_cmap[1],
    )

    #Generate plot of approximated ecdf (see function create_approx_ecdf() above)
    approx_cdf = create_approx_ecdf(beta_1, beta_2, non_dim=False).opts(
        color=bebi103.hv.default_categorical_cmap[0]
    )

    #Return an overlay of both plots
    return (true_cdf * approx_cdf).opts(
        legend_offset=(10, 20), legend_position="bottom_right"
    )

def ecdf_vals_plot(df, label = 'labeled', column_label = 'time to catastrophe (s)', title = 'Labeled vs unlabeled microtubules', conf_int = True, label_bool = True):
    if label_bool:
        p = bokeh_catplot.ecdf(
            data=df,
            cats=[label],
            val=column_label,
            style='staircase',
            conf_int = conf_int,
            title = title,
            width = 450,
            height = 350,
        )

        p.legend.location = 'bottom_right'
        p.legend.title = 'Labeled'

        return bokeh.io.show(p)
    
    else:         
        p = bokeh_catplot.ecdf(
        data=df,
        val=column_label,
        style='staircase',
        conf_int = conf_int,
        title = title,
        width = 450,
        height = 350,
    )

    p.legend.location = 'bottom_right'
    p.legend.title = 'Labeled'

    return bokeh.io.show(p)

def mle_path_plot(bs_reps, param_1 = 'α*', param_2 = 'b*', conf = 0.95):
    # plot bootstrap samples, from BE/Bi103a lesson exercises
    points = hv.Points(data=bs_reps, kdims=[param_1, param_2]).opts(
        size=1, alpha=0.5, padding=0.1
    )

    # Get contour line
    xs, ys = bebi103.viz.contour_lines_from_samples(
        bs_reps[:, 0], bs_reps[:, 1], levels=conf
    )

    # Overlay with sample
    out = points * hv.Path(data=(xs[0], ys[0])).opts(color="black")
    
    return out

def mle_confidence_plot(bs_reps, param_1 = 'α*', param_2 = 'b*', conf = 0.95):
    # Package replicates in data frame for plotting, from BE/Bi103a lesson exercises
    df_res = pd.DataFrame(data=bs_reps, columns=["α*", "b*"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = bebi103.viz.corner(
            samples=df_res, pars=[param_1, param_2], show_contours=True, levels=[conf]
        )

    return bokeh.io.show(p)