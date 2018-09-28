#-*- coding: utf-8 -*-
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ---------- Made by pablo Arias @ircam on 11/2015
# ---------- Copyright (c) 2018 CREAM Lab @ IRCAM
# ----------
# ---------- execute different tests on data
# ---------- to make valid statistics
# ----------
# ---------- import sys
# ---------- sys.path.append('/Users/arias/Documents/Developement/Python/')
# ---------- from stat_assumptions import 'the functions you need'
# ----------
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
from scipy import stats

# ------- check_corelation_assumptions
def check_corelation_assumptions(x, y, plot_details = False, p_treshold = 0.05, compute_correlation = True):
	"""
	In this function we test for normality using the shapiro test and for homoscedasticity using the bartlett test
	Remark:
	A sequence homoscedastic if all random variables in the sequence have the same finite variance
	input:
		array : array to check the correlation assumptions on
	output:
		print the validity of each test
		if print_details is True, this function will plot the details of the computing
	"""
	all_asumptions = True
	# 1 - Check for normality
	#x
	print
	print "** Normality Test  **"
	x_t_statistic, x_p_value = stats.shapiro(x)
	y_t_statistic, y_p_value = stats.shapiro(y)

	if plot_details:
		print "Normality results for x are (x_t_statistic, x_p_value ) : " + x_t_statistic, x_p_value 
		print "Normality results for y are (x_t_statistic, x_p_value ) : " + y_t_statistic, y_p_value 

	#Normality of x
	if x_p_value > p_treshold:
		print 'x is normaly distributed'
	else:
		print 'x is NOT normaly distributed'
		all_asumptions = False
	
	#Normality of y
	if y_p_value > p_treshold:
		print 'y is normaly distributed'
	else:
		print 'y is NOT normaly distributed'
		all_asumptions = False


	# 2 - Check for linearity
	#print
	#print "** Linearity Test **"

	
	# 3 - Check for homoscedasticity
	print
	print "** Homoscedasticity Test **"
	h_t_statistic, h_p_value = stats.bartlett(x,y)
	if plot_details:
		print "Bartelet results for x and y are (x_t_statistic, x_p_value ) : " + h_t_statistic, h_p_value
	
	if h_p_value > p_treshold:
		print 'x and y respect homoscedasticity (same variance)'
	else:
		print 'x and y do not respect homoscedasticity'
		print
		all_asumptions = False


	# 4 - Compute corelation
	if compute_correlation:
		return stats.pearsonr(x,y)
	

# ------- find_transformation_for_normality
def find_transformation_for_normality(x, normality_treshold = 0.05):
	"""
	input:
		array : data to which find the transformation to be normal
	
	output:
		Transformation found for the data to be normal, if found...
	"""	
	import numpy as np

	if check_normality(x):
		print "No transformation is needed"

	#log:
	if check_normality(np.log(x), print_details = False, normality_treshold = normality_treshold):
		return np.log
	
	#sqrt:
	if check_normality(np.sqrt(x), print_details = False, normality_treshold = normality_treshold):
		return np.log		

	return None

# ------- check_normality
def check_normality(x, normality_treshold = 0.05, print_details = True):
	x_t_statistic, x_p_value = stats.shapiro(x)
	
	if print_details:
		print "Normality results for x are (x_t_statistic, x_p_value ) : " + str(x_t_statistic), str(x_p_value )

	if x_p_value > normality_treshold:
		if print_details:
			print 'Array is normaly distributed'
		return True
	else:
		if print_details:
			print 'Array is NOT normaly distributed'
		return False


# ------- check_homoscedasticity
def check_homoscedasticity(): 
	return

def get_cohen_d(treatment, control):
	#For t-test, the effect size is:

	m_t = np.mean(treatment) 
	m_c = np.mean(control) 	

	sd_t = np.std(treatment) 
	sd_c = np.std(control)

	n_t = np.len(treatment)
	n_c = np.len(control)

	sd_pooled = np.sqrt((
						((n_t - 1) * np.pow(sd_t,2)) 
							+ 
						((n_c - 1) * np.pow(sd_c,2))
					   )) /  (n_t+n_c)

	d = m_t - m_c / sd_pooled



