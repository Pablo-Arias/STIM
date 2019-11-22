#-*- coding: utf-8 -*-
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ---------- Made by pablo Arias @ircam on 11/2015
# ---------- Copyright (c) 2018 CREAM Lab // CNRS / IRCAM / Sorbonne Universite
# ----------
# ---------- Analyse audio and return sound features
# ---------- to us this don't forget to include these lines before your script:
# ----------
# ---------- import sys
# ---------- sys.path.append('/Users/arias/Documents/Developement/Python/')
# ---------- from audio_analysis import 'the functions you need'
# ----------
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
from scikits.audiolab import aiffwrite, aiffread, wavwrite, wavread, Sndfile, oggread  # scipy.io.wavfile can't read 24-bit WAV
from super_vp_commands import generate_LPC_analysis
from parse_sdif import mean_matrix, mean_formant_from_sdif, formant_from_sdif
from conversions import lin2db, db2lin, get_file_without_path
import os
import pandas as pd
import glob
import numpy as np

# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ----------------- Spectral centroid
# ----------------- 
# --------------------------------------------------------------------#
def centroid(x, axis):
    return np.sum(x*axis) / np.sum(x) # return weighted mean

def get_spectral_centroid(audio_file, window_size = 256, noverlap = 0, plot_specgram = False):
	"""
 	parameters:
		audio_file 	: file to anlayse
		window_size : window size for FFT computing
		plot_specgram : Do youw ant to plot specgram of analysis?

		returns : time series with the spectral centroid and the time
	
	warning : 
		this function only works for mono files
	"""	
	#imports
	import glob
	from scikits.audiolab import wavread, aiffread
	from scipy import signal
	
	#try to read audofile in different formats
	try:
		sound_in, fs, enc = aiffread(audio_file)
	except ValueError:
		sound_in, fs, enc = wavread(audio_file)
	
	#compute gaussian spectrogram
	f, t, Sxx = signal.spectrogram(sound_in       , fs , nperseg = window_size 
	                               , noverlap = noverlap , scaling ='spectrum'
	                               , mode = 'magnitude'
	                              )

	#plot specgram
	if plot_specgram:
	    plt.figure()
	    fig, ax = plt.subplots()
	    plt.pcolormesh(t, f, Sxx, cmap = 'nipy_spectral')
	    ax.axis('tight')
	    plt.ylabel('Frequency [Hz]')
	    plt.xlabel('Time [sec]')
	    plt.title("Normal FFT of audio signal")
	    plt.show()

	centroid_list = []
	for spectrum in np.transpose(Sxx):
	    centroid_list.append(centroid(spectrum, f))
	    
	return t, centroid_list

def get_mean_spectral_centroid_when_sound(audio_file, RMS_threshold = -55, window_size = 512):
	"""
	Get spectral centroid of audio file only when signal level is above amplitude threshold, and average centroid across these regions

 	Parameters:
		audio_file 	: file to anlayse
		RMS_threshold : min value of RMS to know if there is sound
		window_size : window_size for the RMS compute
	
		returns : spectral centroid when sound
	
	Warning : 
		this function only works for mono files
	"""	
	from bisect import bisect_left

	t, centroid_list = get_spectral_centroid(audio_file, window_size = window_size, noverlap = 0, plot_specgram = False)
	intervals 		 = get_intervals_of_sound(audio_file, RMS_threshold = RMS_threshold, window_size = window_size)

	centroid_means = []
	for interval in intervals:
		begin = bisect_left(t, interval[0])
		end   = bisect_left(t, interval[1])

		local_mean = np.nanmean(centroid_list[begin:end])
		centroid_means.append(local_mean)

	return np.nanmean(centroid_means)


def get_intervals_of_sound(audio_file, RMS_threshold = -50, window_size = 512):
	"""
 	parameters:
		audio_file 	: file to anlayse
		RMS_treshold : min value of RMS to know if there is sound
		window_size : window_size for the RMS compute

		returns : time series with the RMS and the time
	
	warning : 
		this function only works for mono files
	"""		

	t, rmss = get_RMS_over_time(audio_file, window_size = window_size)

	inside_sound = False
	intervals = []
	for i in range(len(rmss)):
	    if inside_sound:
	        if rmss[i] < RMS_threshold:
	            end = t[i]
	            intervals.append([begin, end])
	            inside_sound = False                
	    else:
	        if rmss[i] > RMS_threshold: 
	            begin = t[i]
	            inside_sound = True                
	return intervals

def get_RMS_over_time(audio_file, window_size = 1024, in_db = True):
	"""
 	parameters:
		audio_file 	: file to anlayse
		window_size : window size for FFT computing

		returns : time series with the RMS and the time
	
	warning : 
		this function only works for mono files
	"""	
	import glob
	from scikits.audiolab import wavread, aiffread
	from scipy import signal
	import numpy as np

	try:
		sound_in, fs, enc = aiffread(audio_file)
	except ValueError:
		sound_in, fs, enc = wavread(audio_file)

	begin = 0
	values = []
	time_tags = []
	while (begin + window_size) < len(sound_in):
		data = sound_in[begin : begin + window_size]
		time_tag = (begin + (window_size / 2)) / np.float(fs)
		
		values.append(get_rms_from_data(data, in_db = in_db))
		time_tags.append(time_tag)
		begin = begin + window_size

	return time_tags, values

def get_mean_spectral_centroid(audio_file, window_size = 256, noverlap = 0):
	import numpy as np
	
	#compute centroid
	t, centroid = get_spectral_centroid_over_time(audio_file, window_size = 256, noverlap = 0, plot_specgram = False)

	return np.mean(centroid)

def get_spectral_centroid_over_time(audio_file, window_size = 256, noverlap = 0, plot_specgram = False):
	"""
 	parameters:
		audio_file 	  : file to anlayse
		window_size   : window size for FFT computing
		plot_specgram : Do youw ant to plot specgram of analysis?

		returns : time series with the spectral centroid and the time
	
	warning :
		this function only works for mono files
	"""	
	import glob
	from scikits.audiolab import wavread, aiffread
	from scipy import signal

	try:
		sound_in, fs, enc = aiffread(audio_file)
	except ValueError:
		sound_in, fs, enc = wavread(audio_file)

	#compute gaussian spectrogram
	f, t, Sxx = signal.spectrogram(sound_in       , fs , nperseg = window_size 
	                               , noverlap = noverlap , scaling ='spectrum'
	                               , mode = 'magnitude'
	                              )


	#plot specgram
	if plot_specgram:
	    plt.figure()
	    fig, ax = plt.subplots()
	    plt.pcolormesh(t, f, Sxx, cmap = 'nipy_spectral')
	    ax.axis('tight')
	    plt.ylabel('Frequency [Hz]')
	    plt.xlabel('Time [sec]')
	    plt.title("Normal FFT of audio signal")
	    plt.show()

	centroid_list = []
	for spectrum in np.transpose(Sxx):
	    centroid_list.append(centroid(spectrum, f))
	    
	return t, centroid_list

# Extract_ts_of_pitch_praat
def Extract_ts_of_pitch_praat(Fname):
	"""
		Extract pitch time series using praat for the file Fname

		Fname : input file name
	"""
	import subprocess
	import os	
	Fname = os.path.abspath(Fname)

	out = subprocess.check_output(['/Applications/Praat.app/Contents/MacOS/Praat', "--run", os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ts_pitch.praat'), Fname]);

	out = out.splitlines()

	times = []
	f0s = []

	for line in out:
		line = line.split()
		times.append(line[0])
		f0s.append(line[1])

	#print times
	f0s = [np.nan if item == '--undefined--' else float(item) for item in f0s]
	return times, f0s

def get_mean_pitch_praat(Fname):
	"""
		Extract mean pitch using praat for the file Fname
		
		Input: 
			Fname : input file name
	"""	
	times, f0s = Extract_ts_of_pitch_praat(Fname)
	return np.nanmean(f0s)	

# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ----------------- 		All sort of 		----------------------#
# ----------------- Formant related functions	----------------------#
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
def formant_from_audio(audio_file, nb_formants, ana_winsize=512):
	"""
 	parameters:
		audio_file 	: file to anlayse
		nb_formants : number of formants to compute and return

		This function returns an array of pandas data bases with 
		the formants inside (Frequency, Amplitude, Bandwith and Saliance).
	
	warning : 
		this function only works for mono files
	"""

	#handle paths to create sdif analysis file
	if os.path.dirname(audio_file) == "":
		file_tag = get_file_without_path(audio_file)
		formant_analysis = file_tag + "formant.sdif" 
	else:
		file_tag = get_file_without_path(audio_file)
		formant_analysis = os.path.dirname(audio_file)+ "/" + file_tag + "formant.sdif"

	#generate svp sdif analysis and then read formants from sdif
	from super_vp_commands import generate_formant_analysis
	generate_formant_analysis(audio_file, formant_analysis, nb_formants = nb_formants, ana_winsize = ana_winsize)
	formants = formant_from_sdif(formant_analysis)
	
	os.remove(formant_analysis)
	
	return formants


def get_tidy_formants(audio_file, nb_formants=5, ana_winsize=512, add_harmonicity=False):
	"""
	Convert formant_from_audio data to a tidy dataframe
	parameters:
		audio_file 	: file to anlayse
		nb_formants : number of formants to compute and return
		ana_winsize : window size for the LPC estimation
		add_harmonicity: Add harmonicity values to dataframe
	"""	
	formants = formant_from_audio(audio_file, nb_formants, ana_winsize)

	#tidy up formants
	all_formants = pd.DataFrame()
	for cpt in range(1, len(formants)+1):
		formant = formants[cpt-1]
		formant["Formant"] = ["F" + str(cpt) for i in range(len(formant))]
		all_formants = all_formants.append(formant)

	all_formants.index.names = ['time']

	#Add harmonicity row to formant estimation
	if add_harmonicity:
		def add_harm(row, *args):
			from bisect import bisect
			f0times = args[0]
			f0harm = args[1]
			
			formant_time = row["time"]
			
			index = bisect(f0times, formant_time)
			harm = f0harm[index-1]
			return harm

		f0times, f0harm, _ = get_f0(audio_file = audio_file)
		all_formants = all_formants.reset_index()
		all_formants["harmonicity"] = all_formants.apply(add_harm, axis=1, args=(f0times, f0harm,)) 

	#return formants
	return all_formants


def mean_formant_from_audio(audio_file, nb_formants, use_t_env = True, ana_winsize = 512):
	"""
 	parameters:
		audio_file 	: file to anlayse
		nb_formants : number of formants to compute and return
		use_t_env   : Use true envelope for formant analysis

		This function returns an array of pandas data bases with 
		the formants inside (Frequency, Amplitude, Bandwith and Saliance).
	
	warning : 
		this function only works for mono files
	"""

	#Handle paths to create sdif analysis file
	formant_tag = "_formant.sdif" 
	if os.path.dirname(audio_file) == "":
		file_tag = get_file_without_path(audio_file)
		formant_analysis = file_tag + formant_tag

	else:
		file_tag = get_file_without_path(audio_file)
		formant_analysis = os.path.dirname(audio_file)+ "/" + file_tag + formant_tag

	#Generate sdif analysis file
	from super_vp_commands import generate_formant_analysis, generate_tenv_formant_analysis
	if use_t_env:
		generate_tenv_formant_analysis(audio_file, analysis = formant_analysis, nb_formants = 5, wait = True, ana_winsize = ana_winsize)
	else:
		generate_formant_analysis(audio_file, formant_analysis, nb_formants = nb_formants, ana_winsize = ana_winsize)

	#Get mean formant from sdif analysis file
	mean_formants = mean_formant_from_sdif(formant_analysis)
	
	#Return mean formants
	return mean_formants


def get_formant_data_frame(audio_file, nb_formants = 5 ,t_env_or_lpc = "lpc", destroy_sdif_after_analysis = True, ana_winsize = 512):
	"""
 	parameters:
		audio_file_name 	: pandas data frame with a formant
		t_env_or_lpc        : analysis kind
		
		returns a tidy, good looking, dataframe with the formants' time series

	warning : 
		this function only works for mono files
	"""
	import pandas as pd
	from super_vp_commands import generate_formant_analysis, generate_tenv_formant_analysis

	#Perform formant analysis either with lpc or t_env
	analysis_name = "formant_generated_analysis_X2345.sdif"
	if t_env_or_lpc == "lpc":
		generate_formant_analysis(audio_file, analysis = analysis_name, nb_formants = nb_formants, wait = True, ana_winsize = ana_winsize)
	
	elif  t_env_or_lpc == "t_env":
		generate_tenv_formant_analysis(audio_file, analysis = analysis_name, nb_formants = nb_formants, wait = True, ana_winsize = ana_winsize)
	
	else:
		print "t_env_or_lpc should be either t_env or lpc"

	formants   = formant_from_sdif(analysis_name)

	#parse sdif
	frequencies, amplitudes, bandwidths, Saliances = formant_dataframe_to_arrays(formants)

	#formant to data frame
	formants_df = pd.DataFrame()
	formant_names = ["F1", "F2", "F3", "F4", "F5"]

	for i in range(nb_formants):
		formant = pd.DataFrame()
		formant_name = formant_names[i]
	    
		frequency = frequencies[i].to_frame("frequency")
		amplitude = amplitudes[i].to_frame("amplitude")
		bandwidth = bandwidths[i].to_frame("bandwidth")
		salience  = Saliances[i].to_frame("salience")
		
		dfs = [frequency, amplitude, bandwidth, salience ]
		formant = reduce(lambda left,right: pd.merge(left,right,right_index=True, left_index=True), dfs)
		formant["Formant"] = [formant_name for i in range(len(formant))]
		formants_df = formants_df.append(formant)

	formants_df.index = formants_df.index.rename("time")

	#destroy the sdif file generated if specified by parameter
	if destroy_sdif_after_analysis:
		os.remove(analysis_name)

	return formants_df


def formant_dataframe_to_arrays(formants):
	"""
 	parameters:
		formants 	: pandas data frame with a formant
		
		returns the formant Frequency, Amplitude, Bandwith and Saliance
		as separate variables. 


	warning : 
		this function only works for mono files
	"""
	frequency = [] 
	amplitude = []
	bandwith  = []
	Saliance  = []

	for formant in formants:
		frequency.append(formant['Frequency'])
		amplitude.append(formant['Amplitude'])
		bandwith.append(formant['Bw'])
		Saliance.append(formant['Saliance'])

	return frequency, amplitude, bandwith, Saliance 

#get formant time series with praat
def get_formant_ts_praat(audio_file, time_step=0.001, window_size=0.1, nb_formants=5, max_formant_freq=5500, pre_emph=50.0, add_harmonicity=False):
	"""
		parameters:
		audio_file 	: mono audio file PATH to be analysed, you have to give the ABSOLUTE PATH
		time_step   : Time step in seconds
		window_size : Window Length this is useful for praat
		nb_formants : Number of formants to compute
		max_formant_freq : maximum formant frequency to use in the estimation
		Pre emphasis from Hz : this is a praat parameter for formant estimation, check the get formant function praat's documentation
		
		returns 
			Two beautiful dataframes, one with the frequnecies and one with the bandwidths

	"""
	#call praat to do the analysis
	import subprocess
	import pandas as pd
	import os
	
	#Take the absolulte path
	audio_file = os.path.abspath(audio_file)

	#Call sub process
	out = subprocess.check_output(['/Applications/Praat.app/Contents/MacOS/Praat'
		, "--run"
		, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ts_formants.praat')
		, audio_file
		, str(time_step)
		, str(nb_formants)
		, str(max_formant_freq)
		, str(window_size)
		, str(pre_emph)
		]);

	content = out.splitlines()

	#parse and organize praat's output
	freqs_df = pd.DataFrame()
	bws_df = pd.DataFrame()
	for line in content:
		line = [ np.nan if x == '--undefined--' else np.float(x) for x in line.split()]
		
		frequency = {}
		bandwidth = {}
		for index in range(0,nb_formants):
			frequency['F'+str(index+1)] = line[index*2+1]
			bandwidth['F'+str(index+1)] = line[index*2+2]
		
		freq_df = pd.DataFrame(frequency, index=[line[0]])
		bw_df   = pd.DataFrame(bandwidth, index=[line[0]])
		
		freqs_df = freqs_df.append(freq_df)
		bws_df   = bws_df.append(bw_df)

	#Clean dataframes
	freqs_df = freqs_df.dropna()
	bws_df   = bws_df.dropna()

	freqs_df = freqs_df.stack().to_frame("Frequency")
	freqs_df.index = freqs_df.index.rename(["time", "Formant"])

	bws_df = bws_df.stack().to_frame("Bandwidth")
	bws_df.index = bws_df.index.rename(["time", "Formant"])

	#Add harmonicity row to formant estimation with super vp if harmonicty flag is on
	if add_harmonicity:
		def add_harm(row, *args):
			from bisect import bisect
			f0times = args[0]
			f0harm = args[1]

			formant_time = row["time"]

			index = bisect(f0times, formant_time)
			harm = f0harm[index-1]
			return harm
		
		#Compute f0 with super vp and take the harmonicity
		f0times, f0harm, _ = get_f0(audio_file = audio_file)

		#Add harmonicity to bandwidth
		bws_df = bws_df.reset_index()
		bws_df["harmonicity"] = bws_df.apply(add_harm, axis=1, args=(f0times, f0harm,))
		bws_df = bws_df.set_index("time")

		#Add harmonicity to frequency
		freqs_df = freqs_df.reset_index()
		freqs_df["harmonicity"] = freqs_df.apply(add_harm, axis=1, args=(f0times, f0harm,))
		freqs_df = freqs_df.set_index("time")

	#Return frequencies and bandwidths dataframes	
	return freqs_df, bws_df

def get_mean_formant_praat_harmonicity(audio_file, time_step=0.001, window_size=0.1, nb_formants=5, max_formant_freq=5500, pre_emph=50.0, harmonicity_threshold=None):
	"""
		get mean formant with praat, but exclucding all values under a certain harmonicity threshold 
		(if harmonicity_threshold is defined, if None, just compute mean of time series)
		It's not recomended to use this function without the harmonicity threshold.
		To keep only harmonic sounds use a threshold > 0.85
	"""

	if not harmonicity_threshold:
		#Get mean formant
		freqs_df, bws_df = get_formant_ts_praat(audio_file, time_step = time_step
												, window_size 		  = window_size
												, nb_formants		  = nb_formants
												, max_formant_freq	  = max_formant_freq
												, pre_emph			  = pre_emph
												, add_harmonicity 	  = False
												)

	else:
		freqs_df, bws_df = get_formant_ts_praat(audio_file, time_step = time_step
												, window_size 		  = window_size
												, nb_formants		  = nb_formants
												, max_formant_freq	  = max_formant_freq
												, pre_emph			  = pre_emph
												, add_harmonicity	  = True
												)

		freqs_df 		 = freqs_df.loc[freqs_df["harmonicity"]>harmonicity_threshold]
		bws_df 			 = bws_df.loc[bws_df["harmonicity"]>harmonicity_threshold]
		

	bws_df 	 = bws_df.reset_index().groupby(["Formant"]).mean().reset_index()
	freqs_df = freqs_df.reset_index().groupby(["Formant"]).mean().reset_index()

	return freqs_df, bws_df

#Extract mean formant with praat
def get_mean_formant_praat(Fname):
	"""
	Fname : file name to analyse
	get mean formants with praat script (mean is computed inside praat)
	"""
	import subprocess
	import os
	Fname = os.path.abspath(Fname)

	script_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'formants_mean.praat')
	x = subprocess.check_output(["/Applications/Praat.app/Contents/MacOS/Praat", "--run", script_name, Fname])
	Title, F1, F2, F3, F4, F5 = x.splitlines()

	return float(F1), float(F2), float(F3), float(F4), float(F5)


#Extract mean formant with praat
def get_mean_tidy_formant_praat(Fname):
	"""
	Fname : file name to analyse
	return mean formant values in dataframe
	"""
	import subprocess
	import os

	nb_formants = 5 # TODO parameterize this to Praat

	formant_tags = ["F"+str(i) for i in range(1, nb_formants+1)]
	formants     = get_mean_formant_praat(Fname)
	data = {'Formant': formant_tags, "Frequency": formants}

	return pd.DataFrame.from_dict(data)

	
def analyse_audio_folder(source_folder
						, speed_of_sound		= 335
						, time_step				= 0.001
						, window_size			= 0.01 
						, nb_formants			= 6
						, nb_formants_fd		= 5
						, max_formant_freq		= 5500
						, pre_emph				= 50.0
						, harmonicity_threshold = None 
						, sc_rms_thresh         = -60
						, sc_ws                 = 512
						, parameter_tag         = None
						, print_transformed_file = False
						):
	"""
	Analyse all the sounds from a folder and returns a data frame with key vocal features
	Inputs:
		source_folder 		  : Folder with all the sounds to analyse, prefer mono files, and wav
		harmonicity_threshold : Threshold to filter formant frequencies
		speed_of_sound		  : speed of sound to compute vocal tract length in m.s-1
		time_step			  : The time step to estimate the formants (this is passed to Praat)
		window_size			  : The window size of the formant analysis IN SECONDS  (this is passed to Praat)
		nb_formants			  : Number of formants to estimate  (this is passed to Praat)
		nb_formants_fd 		  : Number of formants to use in the formant dispersion analysis  (this is passed to Praat)
								nb_formants_fd has to be <= to nb_formants
		max_formant_freq	  : Maximum formant frequency available, this is an important praat parameter 
								(5500 usually works well for female voices)
								Check praat's documentation to know more
		pre_emph			  : pre emphasis : check praat documentation on formant estimation
		sc_rms_thresh		  : Amplitude threshold to remove from sound in spectral centroid analysis (in dB)
		sc_ws 		  		  : Window Size in samples for the spectral centoid analysis
		parameter_tag		  : parameter tag should be composed of two values.
									First, the value of the position in the name file where the tag is
									Second, a dictionary with a dictionary with the corresponding parameters to use for each tag
									For example (0, {F: {'max_formant_freq':5500} , M:{'max_formant_freq':5000} })
		print_transformed_file: for debuging purposes, put this to True to print each analysed file

	Output:
		Beautiful dataframe with all the features

	"""

	df_stimuli = pd.DataFrame()

	#for each file in source folder, get all the features
	for file in glob.glob(source_folder+"/*.wav"):
		if print_transformed_file:
			print file
		#handle name of file
		base = get_file_without_path(file)

		#Handle updated parameter tags
		if parameter_tag:
			index = parameter_tag[0]
			key   = base[index]
			parameters = parameter_tag[1][key]

			#update analysis parameters to match the file tag
			if 'harmonicity_threshold' in parameters.keys(): harmonicity_threshold = parameters['harmonicity_threshold']
			if 'time_step' 			   in parameters.keys(): time_step             = parameters['time_step'] 	
			if 'window_size' 		   in parameters.keys(): window_size           = parameters['window_size'] 	
			if 'nb_formants' 		   in parameters.keys(): nb_formants           = parameters['nb_formants'] 		   
			if 'nb_formants_fd' 	   in parameters.keys(): nb_formants_fd        = parameters['nb_formants_fd'] 	   
			if 'max_formant_freq' 	   in parameters.keys(): max_formant_freq      = parameters['max_formant_freq'] 	   
			if 'pre_emph' 			   in parameters.keys(): pre_emph              = parameters['pre_emph'] 			   


		#Get formant frequencies
		df, db = get_mean_formant_praat_harmonicity(file 
											, time_step        = time_step
											, window_size      = window_size
											, nb_formants      = nb_formants
											, max_formant_freq = max_formant_freq
											, pre_emph         = pre_emph
											, harmonicity_threshold   = harmonicity_threshold
											)


		formants = df["Frequency"].values

		#compute formant dispersion
		fd = 0
		for i in range(1, nb_formants_fd):
			fd = fd + formants[i] - formants[i-1]
		fd = fd / (nb_formants_fd -1)

		#get mean pitch
		pitch_mean = get_mean_pitch_praat(os.path.abspath(file))		

		#Compute vocal tract length following Fitch 1997 formulae
		vtl = speed_of_sound/(2*fd)

		#compute mean centroid
		centroid = get_mean_spectral_centroid_when_sound(file, RMS_threshold = sc_rms_thresh, window_size = sc_ws)

		#create dictionary with values and then append to general dataframe
		aux_dict = {}

		#Add fle and CGS
		aux_dict["File"] = base
		aux_dict["CGS"] = centroid
		
		#Add formant information
		for index in range(0, nb_formants):
			aux_dict["F"+str(index+1)] = formants[index]

		#Add other descriptors
		aux_dict["pitch_mean"]   	= pitch_mean
		aux_dict["formant_disp"] 	= fd
		aux_dict["vtl"]   		 	= vtl
		aux_dict["sound_duration"]  = get_sound_duration(file)
		
		#convert dict to dataframe
		aux_df = pd.DataFrame(aux_dict, index=[cpt])

		#append df to general dataframe
		df_stimuli = df_stimuli.append(aux_df)

	return df_stimuli

def get_formant_dispersion(Fname, harmonicity_threshold = None):
	"""
	Compute and return formant dispersion Following Fitch 1997 JASA Formula

	"""
	#Get formant frequencies
	if harmonicity_threshold:
		df, db = get_mean_formant_praat_harmonicity(Fname, harmonicity_threshold = harmonicity_threshold)
		F1 = df.loc[df["Formant"] == "F1"]["Frequency"].values[0]
		F2 = df.loc[df["Formant"] == "F2"]["Frequency"].values[0]
		F3 = df.loc[df["Formant"] == "F3"]["Frequency"].values[0]
		F4 = df.loc[df["Formant"] == "F4"]["Frequency"].values[0]
		F5 = df.loc[df["Formant"] == "F5"]["Frequency"].values[0]

	else:
		F1, F2, F3, F4, F5 = get_mean_formant_praat(os.path.abspath(Fname))


	#compute formant_dispersion
	fd = (F2 - F1 + F3 - F2 + F4 - F3 + F5 - F4)  / 4

	return fd

def estimate_vtl(Fname, harmonicity_threshold=None, speed_of_sound=335):
	"""
	Estimate vocal tract length following the formula in Fitch 1997
	the speed_of_sound is usually ~335 m/s, and L is the vocal tract length in m.
	"""
	
	fd = get_formant_dispersion(Fname=Fname, harmonicity_threshold=harmonicity_threshold)

	vtl = speed_of_sound/(2*fd)

	return vtl


# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ----------------- Spectral Enveloppes
# ----------------- LPC and True Envelope
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#

# ---------- get_mean_lpc_from_audio
def get_mean_lpc_from_audio(audio,wait = True, nb_coefs = 45, destroy_sdif_after_analysis = True):
 	"""
 	parameters:
		audio 		: file to anlayse
		analysis 	: sdif file to generate, if not defined analysis will have the same name as the audio file but with an sdif extension.
		nb_coefs 	: number of lpc coeficients for the analysis

	warning : 
		this function only works for mono files
	"""
	path_and_name = os.path.basename(audio)
	path_and_name = os.path.splitext(path_and_name)[0]
	analysis = path_and_name + ".sdif"

	# first generate an sdif analysis file 
	generate_LPC_analysis(audio_file = audio, analysis = analysis, wait = True, nb_coefs = nb_coefs)
	
	# then load and mean the sdif file generated
	matrix_data =  mean_matrix(analysis)

	#destroy the sdif file generated if specified by parameter
	if destroy_sdif_after_analysis:
		os.remove(analysis)

	return matrix_data


# ---------- get_true_env_analysis
def get_true_env_analysis(audio_file, analysis="", wait = True, f0 = "mean", destroy_sdif_after_analysis = True):
	"""
    Get mean true env data

    return:
		spectral envelope
    """
	if analysis== "":
		if os.path.dirname(audio_file) == "":
			file_tag = os.path.basename(audio_file)
			file_tag = os.path.splitext(file_tag)[0]
			analysis = file_tag + ".sdif" 
		else:
			file_tag = os.path.basename(audio_file)
			file_tag = os.path.splitext(file_tag)[0]
			analysis = os.path.dirname(audio_file)+ "/" + file_tag + ".sdif"

	from parse_sdif import mean_matrix
	from super_vp_commands import generate_true_env_analysis

	generate_true_env_analysis(audio_file, analysis=analysis, wait = wait, f0 = f0)

	matrix_data = mean_matrix(analysis)

	#destroy the sdif file generated if specified by parameter
	if destroy_sdif_after_analysis:
		os.remove(analysis)

	return matrix_data

# ---------- get_true_env_analysis
def get_true_env_ts_analysis(audio_file, analysis="", wait = True, f0 = "mean", destroy_sdif_after_analysis = True):
	"""
    Get true env time series data
    """
	if analysis== "":
		if os.path.dirname(audio_file) == "":
			file_tag = os.path.basename(audio_file)
			file_tag = os.path.splitext(file_tag)[0]
			analysis = file_tag + ".sdif" 
		else:
			file_tag = os.path.basename(audio_file)
			file_tag = os.path.splitext(file_tag)[0]
			analysis = os.path.dirname(audio_file)+ "/" + file_tag + ".sdif"

	from parse_sdif import get_matrix_values
	from super_vp_commands import generate_true_env_analysis

	generate_true_env_analysis(audio_file, analysis=analysis, wait = wait, f0 = f0)

	tlist, matrix_data = get_matrix_values(analysis);

	#destroy the sdif file generated if specified by parameter
	if destroy_sdif_after_analysis:
		os.remove(analysis)

	return tlist, matrix_data



# ---------- get_mean_spectre
def get_spectrum(file, window_size = 256, noverlap = 0, plot_spec = False, plot_mean_spec = False):
	"""
    Get spectrum of file and plot it if needed
    plot_spec : plot spectrum
    plot_mean_spec : plot mean spectrum over time
    return:
    	f : frequency bins
    	t : time bins
    	Sxx : FFT matrix
    """
	import glob
	from scikits.audiolab import wavread
	from scipy import signal
	import matplotlib.pyplot as plt

	sound_in, fs, enc = wavread(file)
	#compute gaussian spectrogram
	f, t, Sxx = signal.spectrogram(sound_in       , fs , nperseg = window_size 
	                               , noverlap = noverlap , scaling ='spectrum'
	                               , mode = 'magnitude'
	                              )
	if plot_spec:
	    plt.figure()
	    fig, ax = plt.subplots()
	    plt.pcolormesh(t, f, Sxx, cmap = 'nipy_spectral')
	    ax.axis('tight')
	    plt.ylabel('Frequency [Hz]')
	    plt.xlabel('Time [sec]')
	    plt.title("Normal FFT of audio signal")
	    plt.show()

	mean_sxx = np.mean(Sxx, axis = 1)
	if plot_mean_spec:
	    plt.figure()
	    fig, ax = plt.subplots()
	    plt.semilogx(f,  mean_sxx)
	    ax.axis('tight')
	    plt.xlabel('Amplitude')
	    plt.xlabel('Frequency [Hz]')
	    plt.title("Mean spectrum of audio signal")
	    plt.show()        


	return f, t, Sxx



# ---------- get_f0
def get_f0(audio_file, analysis ="",wait = True, destroy_sdif_after_analysis = True):
	"""
    Get f0
    	audio_file : file to analyse

    return : 
    	f0times: time tags
    	f0harm: signal harmonicity
    	f0val: fundamental frequency estimation in Hz

    """
	if os.path.dirname(audio_file) == "":
		file_tag = os.path.basename(audio_file)
		file_tag = os.path.splitext(file_tag)[0]
		f0_analysis = file_tag + "f0.sdif" 
		if analysis == "":
			analysis = file_tag + ".sdif" 
	else:
		file_tag = os.path.basename(audio_file)
		file_tag = os.path.splitext(file_tag)[0]
		f0_analysis = os.path.dirname(audio_file)+ "/" + file_tag + "f0.sdif"
	
	from super_vp_commands import generate_f0_analysis
	generate_f0_analysis( audio_file, f0_analysis)

	
	from parse_sdif import get_f0_info
	f0times, f0harm, f0val = get_f0_info(f0_analysis)
	

	#destroy the sdif file generated if specified by parameter
	if destroy_sdif_after_analysis:
		os.remove(f0_analysis)

	return f0times, f0harm, f0val



# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ----------------- Dynamics
# ----------------- 
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#

# ---------- Compute Zero Cross Rate from data
def get_zero_cross_from_data(sound_data, window_size) :
	index = 0
	zero_cross = []
	while index < (len(sound_data) - window_size) / window_size:
		begin = index * window_size
		end = begin + window_size
		data_to_analyse = sound_data[begin:end]
		#count the number of 0s in sound data between beg and end 
		zero_cross.append(sum(x > 0 for x in data_to_analyse))

		index = index + 1

	#norm
	zero_cross = [float(i)/max(zero_cross) for i in zero_cross]

	return zero_cross

def get_rms_from_wav(audio_file):
    """
    Returns the root-mean-square (power) of the audio buffer
    """
    from scikits.audiolab import wavread

    data, fs, enc = wavread(audio_file)
    return get_rms_from_data(data)

def get_rms_from_data(data, in_db = True):
    """
    Returns the root-mean-square (power) of the audio buffer
    """
    from conversions import lin2db
    rms = np.sqrt(np.mean(data**2))
    if in_db:
	    rms 		= lin2db(rms)
    return rms



# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ----------------- Others - attributes
# ----------------- 
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
def get_sound_duration(file):
	"""
	returns sound duration in seconds
	"""
	from scikits.audiolab import wavread
	sound_in, sr, pcm = wavread(file)

	return len(sound_in)/float(sr)

def get_nb_channels(audio_file):
	"""
	get number of channels of audiofile
	"""
	f = Sndfile(audio_file, 'r')
	return f.channels	


def find_silence(audio_file, threshold = -65, wnd_size = 16384):
	"""
	find a segment of silence (<threshold dB)in the sound file
	return tag in seconds
	"""	
	try:
		x, fs, enc 		= aiffread(str(audio_file))
	except:
		x, fs, enc 		= wavread(str(audio_file))

	index = 0
	NbofWrittendFiles = 1
	silence_tags = []
	while index + wnd_size < len(x):
 		DataArray 	= x[index: index + wnd_size]
 		rms 		= np.sqrt(np.mean(np.absolute(DataArray)**2))
 		rms 		= lin2db(rms)
		index 		= wnd_size + index
		if rms < threshold:
			end 		= 0
			begining 	= index
			index 		= wnd_size + index
			while rms < threshold:
				if index + wnd_size < len(x):
					index 		= wnd_size + index
					DataArray 	= x[index: index + wnd_size]
 					rms 		= np.sqrt(np.mean(np.absolute(DataArray)**2))
 					rms 		= lin2db(rms)
 					end 		= index
 				else:
 					break

			
 			#if file is over 250 ms long, write it
 			if (end - begining) > (fs / 8) :
 				begining = begining - wnd_size
 				if begining < 0: 
 					begining = 0
 				 
 				end = end + wnd_size
 				if end > len(x): 
 					end = len(x)

 				#samples to seconds, minutes, hours 
 				begining_s = begining/float(fs)
 				end_s = end/float(fs)
 				silence_tags.append([begining_s, end_s])

 	return silence_tags