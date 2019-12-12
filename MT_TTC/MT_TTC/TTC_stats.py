""""Contains calculations of data analysis that is agnostic of a model distribution
    Methods included:
    
        bs_reps:
        Returns bootstrap replicates of a given size from a dataset 
        - arguments:
            data: The data to take bootstrap replicates from
            size: size of the bootstrap samples
            
        draw_bs_reps_mean:
        Creates 'size' mounts of boostrap replicates and returns their mean in an array.
        - arguments:
            data: The data to take bootstrap replicates from
            size: size of the bootstrap replicates
            
        mean_and_confidence:
        Takes a data frame or numpy array as input and gives the mean time to catastrophe with and a confidence interval
        - arguments:
              df: data frame to calculate the mean and the confidence from
        
        TTC_p_value:
        Gives p_value for the null hypothesis that two distributions are exactly the same.
        - arguments:
            df_1: data frame from the first distribution to be tested
            df_2: data frame from the second distribution to be tested
            mean_1: mean of the first distribution
            mean_2: mean of the second distribution
			
		df_TTC:
		Returns table with the means and confidence intervals of TTC data (labeled and unlabeled)
        """

#Preamb

#Data analysis
import pandas as pd
import scipy.special
from scipy import stats
import numpy as np
import math
import copy


#Necessary for 
# Draw n_samples from bootstrap samples
def bs_reps(data, size=1):
    out = np.array([np.random.choice(data, size=len(data)) for i in range(size)])
    return out


# Get the mean from bootstrap reps
def draw_bs_reps_mean(data, size=1):
    out = np.empty(size)
    for i in range(size):
        out[i] = np.mean(bs_reps(data))
    return out

def mean_and_confidence(df, column = 'time to catastrophe (s)'):

	# Number of bootstrap samples to be used 
	n_samples = len(df)

	# Get arithmetic means of the data
	mean = np.mean(df[column].values)

	#Draw bootstrap samples and get the means for the bootstrap reps
	mean_bs_reps = draw_bs_reps_mean(df[column].values, size = n_samples)

	# Calculate the confidence intervals
	mean_conf_int = np.percentile(mean_bs_reps, [2.5, 97.5])


	return mean, mean_conf_int


def TTC_p_value(df, mean_1, mean_2, size = 1, absolute=True, col_label = 'time to catastrophe (s)'):

	perm_reps = np.empty(size)
	# Create 'size' permutations of the data and get the difference of means of the new split datasets.
	for i in range(size):
		all_data = copy.deepcopy(df[col_label].values)
		np.random.shuffle(all_data)
		x_data = all_data[:size]
		y_data = all_data[size:]
		if absolute == True:
			perm_reps[i] = abs(np.mean(x_data) - np.mean(y_data))
		else:
			perm_reps[i] = np.mean(x_data) - np.mean(y_data)
			
	diff_mean = abs(mean_1 - mean_2)
	#Calculate p-value
	p_val = np.sum(perm_reps >= diff_mean) /len(perm_reps)

	return p_val
	
	
def df_TTC(df_labeled, df_unlabeled):

	#Get the means and confidence intervals
	labeled_mean = mean_and_confidence(df_labeled)
	unlabeled_mean = mean_and_confidence(df_unlabeled)
	indexes = [True, False]
	means = np.array([round(labeled_mean[0],2), round(unlabeled_mean[0],2)])
	lower_bounds = np.array([round(labeled_mean[1][0],2), round(unlabeled_mean[1][0],2)])
	upper_bounds = np.array([round(labeled_mean[1][1],2), round(unlabeled_mean[1][1],2)])
	df = pd.DataFrame({'GFP label': indexes, 'Mean (s)': means, 'Lower bound': lower_bounds, 'Upper bound': upper_bounds})
	
	return df