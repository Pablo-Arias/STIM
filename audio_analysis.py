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
from conversions import lin2db, db2lin
import os
import pandas as pd


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

def get_mean_spectral_centroid_when_sound(audio_file, RMS_threshold = -55, window_size = 512):
	"""
	Get spectral centroid of audio file only when there is sound and mean across these regions

 	parameters:
		audio_file 	: file to anlayse
		RMS_threshold : min value of RMS to know if there is sound
		window_size : window_size for the RMS compute
	
		returns : spectral centroid when sound
	
	warning : 
		this function only works for mono files
	"""	
	from bisect import bisect_left

	t, centroid_list = get_spectral_centroid(audio_file, window_size = window_size, noverlap = 0, plot_specgram = False)
	intervals = get_intervals_of_sound(audio_file, RMS_threshold = RMS_threshold, window_size = window_size)

	centroid_means = []
	for interval in intervals:
		begin = bisect_left(t, interval[0])
		end = bisect_left(t, interval[1])

		local_mean = np.nanmean(centroid_list[begin:end])
		centroid_means.append(local_mean)

	return np.mean(centroid_means)


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
		audio_file 	: file to anlayse
		window_size : window size for FFT computing
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

# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ----------------- Formants
# ----------------- 
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

	if os.path.dirname(audio_file) == "":
		file_tag = os.path.basename(audio_file)
		file_tag = os.path.splitext(file_tag)[0]
		formant_analysis = file_tag + "formant.sdif" 
	else:
		file_tag = os.path.basename(audio_file)
		file_tag = os.path.splitext(file_tag)[0]
		formant_analysis = os.path.dirname(audio_file)+ "/" + file_tag + "formant.sdif"

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

	#Tidy formants
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
	formant_tag = "_formant.sdif" 
	if os.path.dirname(audio_file) == "":
		file_tag = os.path.basename(audio_file)
		file_tag = os.path.splitext(file_tag)[0]
		formant_analysis = file_tag + formant_tag
	else:
		file_tag = os.path.basename(audio_file)
		file_tag = os.path.splitext(file_tag)[0]
		formant_analysis = os.path.dirname(audio_file)+ "/" + file_tag + formant_tag

	from super_vp_commands import generate_formant_analysis, generate_tenv_formant_analysis
	
	if use_t_env:
		generate_tenv_formant_analysis(audio_file, analysis = formant_analysis, nb_formants = 5, wait = True, ana_winsize = ana_winsize)
	else:
		generate_formant_analysis(audio_file, formant_analysis, nb_formants = nb_formants, ana_winsize = ana_winsize)

	mean_formants = mean_formant_from_sdif(formant_analysis)
	
	#os.remove(formant_analysis)
	
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

	#do formant analysis
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
		formant 	: pandas data frame with a formant
		
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

import numpy as np

# Extract_ts_of_pitch_praat
def Extract_ts_of_pitch_praat(Fname):
	"""
		Fname : input file name
	"""
	import subprocess
	import os	

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


#get formant time series with praat
def get_formant_ts_praat(audio_file):
	"""
		parameters:
		audio_file 	: mono audio file PATH to be analysed, you have to give the ABSOLUTE PATH
		
		returns 
			Two beutiful dataframes, one with the frequnecies and one with the bandwidths

	"""
	#call praat to do the analysis
	import subprocess
	import pandas as pd
	import os
	audio_file = os.path.abspath(audio_file)
	
	# cmd = [‘/Applications/Praat.app/Contents/MacOS/Praat’
 #                    , os.path.join(os.path.dirname(os.path.realpath(__file__)), ‘ts_formants.praat’)
 #                    , audio_file
 #         ]
 #    content = subprocess.check_output(cmd, shell=False, stderr=subprocess.STDOUT).splitlines()

	out = subprocess.check_output(['/Applications/Praat.app/Contents/MacOS/Praat', "--run", os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ts_formants.praat'), audio_file]);
	content = out.splitlines()

	#parse and organize praat's output
	freqs_df = pd.DataFrame()
	bws_df = pd.DataFrame()
	for line in content:
		line = [ np.nan if x == '--undefined--' else np.float(x) for x in line.split()]
		frequency = { 'F1' : line[1] ,'F2' : line[2] ,'F3' : line[3] ,'F4' : line[4] , 'F5' : line[5] }
		bandwidth = {'F1'  : line[6], 'F2' : line[7] ,'F3' : line[8] ,'F4' : line[9] , 'F5' : line[10]}
		
		freq_df = pd.DataFrame(frequency, index=[line[0]])
		bw_df   = pd.DataFrame(bandwidth, index=[line[0]])
		
		freqs_df = freqs_df.append(freq_df)
		bws_df   = bws_df.append(bw_df)

	
	freqs_df = freqs_df.dropna()
	bws_df   = bws_df.dropna()

	freqs_df = freqs_df.stack().to_frame("Frequency")
	freqs_df.index = freqs_df.index.rename(["time", "Formant"])

	bws_df = bws_df.stack().to_frame("Bandwidth")
	bws_df.index = bws_df.index.rename(["time", "Formant"])

	return freqs_df, bws_df


#Extract mean formant with praat
def get_mean_formant_praat(Fname):
	"""
	Fname : file name to analyse
	get mean formants with praat
	"""
	import subprocess
	import os

	script_name=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'formants_mean.praat')
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
def get_f0(audio_file, f0_analysis ="", wait = True, destroy_sdif_after_analysis = True):
	"""
    Get f0
    	audio_file : file to analyse

    return : 
    	f0times: time tags
    	f0harm:  signal harmonicity
    	f0val:   fundamental frequency estimation in Hz
    """

	if f0_analysis == "":
		file_tag = os.path.basename(audio_file)
		file_tag = os.path.splitext(file_tag)[0]
		file_tag = file_tag + "_f0.sdif"
		f0_analysis = os.path.join(os.path.dirname(audio_file), file_tag)
		 
	from super_vp_commands import generate_f0_analysis
	generate_f0_analysis(audio_file, f0_analysis)
	
	from parse_sdif import get_f0_info
	f0times, f0harm, f0val = get_f0_info(f0_analysis)
	
	#destroy the sdif file generated if specified by parameter
	if destroy_sdif_after_analysis:
		os.remove(f0_analysis)

	return f0times, f0harm, f0val

def get_mean_f0(audio_file, max_harm=0.5, plot=False):
	"""
	Function to get mean F0 of an audio file as a float in Hz
	INPUT
		audio_file : wav file to analyse
		max_harm   : harmonicity threshold under which values will be replaced by NaN
		plot       : plots F0 in function of time
	returns :
		Mean F0 in Hz (float)
	"""
	import pandas as pd
	import matplotlib.pylab as plt
	f0times,f0harm,f0val = get_f0(audio_file = audio_file)
	f0_audio = pd.DataFrame()
	f0_audio["harm"] = f0harm
	f0_audio["f0val"] = f0val
	f0_audio["time"] = f0times
	f0_audio[f0_audio.harm < max_harm] = np.nan
	if plot:
		plt.plot(f0_audio.time, f0_audio.f0val)
		plt.xlabel('time [s]')
		plt.ylabel('F0 [Hz]')
	return f0_audio.f0val.mean()

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