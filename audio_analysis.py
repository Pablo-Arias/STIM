#-*- coding: utf-8 -*-
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ---------- Made by pablo Arias @ircam on 11/2015
# ---------- Copyright (c) 2018 CREAM Lab // CNRS / IRCAM / Sorbonne Universite
# ----------
# ---------- Analyse audio and return sound features
# ----------
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
from __future__ import absolute_import
from __future__ import print_function
import os
import glob
from bisect import bisect

import soundfile
import pandas as pd
import numpy as np

from super_vp_commands import generate_LPC_analysis
from conversions import lin2db, get_file_without_path
from six.moves import range
from functools import reduce
import contextlib
import collections
import parselmouth
from parselmouth.praat import call

# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ----------------- Spectral centroid
# ----------------- 
# --------------------------------------------------------------------#
def centroid(x, axis):
	sum_x = np.sum(x)
	if sum_x == 0:
		return np.nan  # Return NaN or an appropriate value to handle the divide-by-zero case
	return np.sum(x * axis) / sum_x

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
	from scipy import signal
	
	#Read audio file
	sound_in, fs = soundfile.read(audio_file)
	
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
	from scipy import signal
	import numpy as np

	#Read audio file
	sound_in, fs = soundfile.read(audio_file)

	begin = 0
	values = []
	time_tags = []
	while (begin + window_size) < len(sound_in):
		data = sound_in[begin : begin + window_size]
		time_tag = (begin + (window_size / 2)) / float(fs)
		
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
	from scipy import signal

	#Read audio file
	sound_in, fs = soundfile.read(audio_file)

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
def Extract_ts_of_pitch_praat(Fname
							, time_step=0.001 
							, pitch_floor = 75
							, pitch_ceiling =  350
							, max_nb_candidates=15
							, accuracy = 0
							, silence_threshold = 0.03
							, voicing_threshold = 0.45
							, octave_cost = 0.01
							, octave_jump_cost = 0.35 
							, voiced_unvoiced_cost = 0.14
							, harmonicity_threshold = None
							):
	"""
		Extract pitch time series using praat for the file Fname

		Input: 
			Fname : input file name
			time_step : time step to use for the analysis in seconds
			pitch_floor : minimul pitch posible, in HZ
			pitch_ceiling : maximum pitch posible, in HZ
			accuracy : on or off
		
		See parameters here : https://www.fon.hum.uva.nl/praat/manual/Sound__To_Pitch__raw_ac____.html

		Also, use harmonicity threshold (from 0 to 1) to clean pitch estimation values.
		Return: times, f0

	"""
	import subprocess
	import os
	import parselmouth
	Fname = os.path.abspath(Fname)

	#execture script with parselmouth
	script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ts_pitch.praat')
	_, out = parselmouth.praat.run_file(script
										, Fname
										, str(time_step)
										, str(pitch_floor)
										, str(max_nb_candidates)
										, str(accuracy)
										, str(silence_threshold)
										, str(voicing_threshold)
										, str(octave_cost)
										, str(octave_jump_cost)
										, str(voiced_unvoiced_cost)
										, str(pitch_ceiling)
										, capture_output=True
										)

	#Parse script
	out = out.splitlines()
	times = []
	f0s = []

	for line in out:
		line = line.split()
		times.append(line[0])
		f0s.append(line[1])

	f0s = [np.nan if item == u'--undefined--' else float(item) for item in f0s]

	#If harmonicity threshold is defined, clean with harm thresh
	if harmonicity_threshold:
		times =  [float(x) for x in times]
		f0s   = [float(x) for x in f0s]
		
		from bisect import bisect_left
		harm_time, harm_vals = get_harmonicity_ts(Fname, time_step=time_step, normalise=True)

		cleaned_vals  = []
		cleaned_times = []
		for index, time in enumerate(times):
			idx_harm = bisect_left(harm_time, time)

			
			if harmonicity_threshold < harm_vals[idx_harm-1]:
				cleaned_vals.append(f0s[index])
				cleaned_times.append(time)

		
		times, f0s = cleaned_times, cleaned_vals

	return times, f0s

def get_mean_pitch_praat(Fname, time_step=0.001 , pitch_floor = 75, pitch_ceiling =  350, harmonicity_threshold = None):
	"""
		Extract mean pitch using praat for the file Fname
		
		Input: 
			time_step : time step to use for the analysis in seconds - 0.0 : auto
			pitch_floor : minimul pitch posible, in HZ
			pitch_ceiling : maximum pitch posible, in HZ
		
		Output:
			mean pitch

	"""	
	times, f0s = Extract_ts_of_pitch_praat(Fname, time_step=time_step , pitch_floor = pitch_floor, pitch_ceiling =  pitch_ceiling, harmonicity_threshold=harmonicity_threshold)
	return np.nanmean(f0s)

def get_pitch_std(Fname, time_step=0.001 , pitch_floor = 75, pitch_ceiling =  350, harmonicity_threshold=None):

	times, f0s  = Extract_ts_of_pitch_praat(Fname=Fname, time_step=time_step , pitch_floor = pitch_floor, pitch_ceiling =  pitch_ceiling, harmonicity_threshold=harmonicity_threshold)

	return np.nanstd(f0s)


def get_pulse_features(sound, f0min=75, f0max=450, time_step=0.01, silence_threshold=0.1, periods_per_window=1.0, start=0, stop=0, period_floor=0.0001, period_ceiling = 0.02, max_period_factor=1.3):
	
	
	shim_df = get_shimmer(sound
	                      , f0min = f0min
	                      , f0max = f0max
	                     )

	hnr_df  = get_hnr(sound
	                  , time_step          = time_step
	                  , f0min              = f0min
	                  , silence_threshold  = silence_threshold
	                  , periods_per_window = periods_per_window
	                  , start              = start
	                  , stop               = stop
	                 )

	jitt_df = get_jitter(sound
	                     , f0min  = f0min
	                     , f0max  = f0max
	                     , start  = start
	                     , stop   = stop
	                     , period_floor   = period_floor
	                     , period_ceiling = period_ceiling
	                     , max_period_factor = max_period_factor
	                    )
	
	pulses_df = jitt_df.merge(shim_df    , on ="sound")
	pulses_df = pulses_df.merge(hnr_df   , on ="sound")
	return pulses_df

# get_harmonicity_ts
def get_harmonicity_ts(file, time_step, min_pitch = 75, silence_threshold = 0.1, periods_per_window = 4.5, normalise=True):
	"""
	Use Parselmouth to extract harmonicity of file
	See parameter explanation here : https://www.fon.hum.uva.nl/praat/manual/Sound__To_Harmonicity__ac____.html
	"""
	import numpy as np
	import parselmouth
	from parselmouth.praat import call

	#Extract harmonicity with parselmouth
	sound = parselmouth.Sound(file)
	harm_obj = call(sound, "To Harmonicity (cc)...", time_step, min_pitch, silence_threshold, periods_per_window)
	duration = harm_obj.duration
	harm_time = np.arange(0, duration, time_step)
	harm_vals = [harm_obj.get_value(i) for i in harm_time]

	if normalise:
		#Normalise between 0 and 1 using -200 as the lowest value
		for cpt in range(len(harm_vals)):
			if harm_vals[cpt] < -200:
				harm_vals[cpt] = -200
			if harm_vals[cpt] > 1:
				harm_vals[cpt] = 1

		for cpt in range(len(harm_vals)):
			harm_vals[cpt] = (harm_vals[cpt] + 201) / 201

	return harm_time, harm_vals

#from : https://osf.io/qe9k4/
def get_jitter(sound, f0min=75, f0max=450, start=0, stop=0, period_floor=0.0001, period_ceiling=0.02, max_period_factor=1.3):
	"""
	start and stop = 0 means analyse all sound. In seconds.
	period floor   : In seconds. The shortest possible interval that will be used in the computation of jitter, in seconds. 
	If an interval is shorter than this, it will be ignored in the computation of jitter (and the previous and next intervals will not be regarded as consecutive). 
	This setting will normally be very small, say 0.1 ms.

	period ceiling : In seconds. The longest possible interval that will be used in the computation of jitter, in seconds. 
	If an interval is longer than this, it will be ignored in the computation of jitter (and the previous and next intervals will not be regarded as consecutive). For example, if the minimum frequency of periodicity is 50 Hz, set this setting to 0.02 seconds; intervals longer than that could be regarded as voiceless stretches and will be ignored in the computation.

	Maximum period factor : the largest possible difference between consecutive intervals that will be used in the computation of jitter. 
	If the ratio of the durations of two consecutive intervals is greater than this, this pair of intervals will be ignored in the computation of jitter (each of the intervals could still take part in the computation of jitter in a comparison with its neighbour on the other side).


	Check the documentation here : https://www.fon.hum.uva.nl/praat/manual/PointProcess__Get_jitter__local____.html
	"""
	df_aux = pd.DataFrame()
	praat_sound         = parselmouth.Sound(sound) # read the sound
	pointProcess        = call(praat_sound, "To PointProcess (periodic, cc)", f0min, f0max)
	localJitter         = call(pointProcess, "Get jitter (local)"           , start, stop, period_floor, period_ceiling, max_period_factor)
	localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)" , start, stop, period_floor, period_ceiling, max_period_factor)
	rapJitter           = call(pointProcess, "Get jitter (rap)"             , start, stop, period_floor, period_ceiling, max_period_factor)
	ppq5Jitter          = call(pointProcess, "Get jitter (ppq5)"            , start, stop, period_floor, period_ceiling, max_period_factor)
	ddpJitter           = call(pointProcess, "Get jitter (ddp)"             , start, stop, period_floor, period_ceiling, max_period_factor)

	#Collect data
	df_aux["sound"]                        = [sound]
	df_aux["localJitter"]                  = [localJitter]
	df_aux["localabsoluteJitterrapJitter"] = [localabsoluteJitter]
	df_aux["rapJitter"]                    = [rapJitter]
	df_aux["ppq5Jitter"]                   = [ppq5Jitter]
	df_aux["ddpJitter"]                    = [ddpJitter]

	return df_aux

def get_hnr(sound, time_step=0.01, f0min=75, silence_threshold=0.1, periods_per_window=1.0, start=0, stop=0):
	"""
	if both start and stop = 0, compute descriptor for all sound
	"""
	praat_sound       = parselmouth.Sound(sound) # read the sound
	harmonicity = call(praat_sound, "To Harmonicity (cc)", time_step, f0min, silence_threshold, periods_per_window)
	hnr         = call(harmonicity, "Get mean"              , start, stop)
	
	df_aux = pd.DataFrame()
	df_aux["sound"] = [sound]
	df_aux["hnr"] = [hnr]

	return df_aux    
    
def get_shimmer(sound, f0min = 74, f0max = 350):
	praat_sound          = parselmouth.Sound(sound) # read the sound
	pointProcess        = call(praat_sound, "To PointProcess (periodic, cc)", f0min, f0max)
	localShimmer        = call([praat_sound, pointProcess], "Get shimmer (local)"    , 0, 0, 0.0001, 0.02, 1.3, 1.6)
	localdbShimmer      = call([praat_sound, pointProcess], "Get shimmer (local_dB)" , 0, 0, 0.0001, 0.02, 1.3, 1.6)
	apq3Shimmer         = call([praat_sound, pointProcess], "Get shimmer (apq3)"     , 0, 0, 0.0001, 0.02, 1.3, 1.6)
	aqpq5Shimmer        = call([praat_sound, pointProcess], "Get shimmer (apq5)"     , 0, 0, 0.0001, 0.02, 1.3, 1.6)
	apq11Shimmer        = call([praat_sound, pointProcess], "Get shimmer (apq11)"    , 0, 0, 0.0001, 0.02, 1.3, 1.6)
	ddaShimmer          = call([praat_sound, pointProcess], "Get shimmer (dda)"      , 0, 0, 0.0001, 0.02, 1.3, 1.6)

	df_aux = pd.DataFrame()
	df_aux["sound"]          = [sound]
	df_aux["localShimmer"]   = [localShimmer]
	df_aux["localdbShimmer"] = [localdbShimmer]
	df_aux["apq3Shimmer"]    = [apq3Shimmer]
	df_aux["aqpq5Shimmer"]   = [aqpq5Shimmer]
	df_aux["apq11Shimmer"]   = [apq11Shimmer] 
	df_aux["ddaShimmer"]     = [ddaShimmer]

	return df_aux

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
	from parse_sdif import formant_from_sdif
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
		#all_formants = all_formants.append(formant)
		all_formants = pd.concat([all_formants, formant])

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
		
		harm_time, harm_vals = get_harmonicity_ts(audio_file, time_step=time_step, normalise=True)

		all_formants = all_formants.reset_index()
		all_formants["harmonicity"] = all_formants.apply(add_harm, axis=1, args=(harm_time, harm_vals,)) 

	#return formants
	return all_formants


def mean_formant_from_audio(audio_file, nb_formants, use_t_env = True, ana_winsize = 512, destroy_sdif_after_analysis = True):
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
	from parse_sdif import mean_formant_from_sdif
	if use_t_env:
		generate_tenv_formant_analysis(audio_file, analysis = formant_analysis, nb_formants = 5, wait = True, ana_winsize = ana_winsize)
	else:
		generate_formant_analysis(audio_file, formant_analysis, nb_formants = nb_formants, ana_winsize = ana_winsize)

	#Get mean formant from sdif analysis file
	mean_formants = mean_formant_from_sdif(formant_analysis)

	#destroy the sdif file generated if specified by parameter
	if destroy_sdif_after_analysis:
		os.remove(formant_analysis)
	
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
	from super_vp_commands import generate_formant_analysis, generate_tenv_formant_analysis
	from parse_sdif import formant_from_sdif

	#Perform formant analysis either with lpc or t_env
	analysis_name = "formant_generated_analysis_X2345.sdif"
	if t_env_or_lpc == "lpc":
		generate_formant_analysis(audio_file, analysis = analysis_name, nb_formants = nb_formants, wait = True, ana_winsize = ana_winsize)
	
	elif  t_env_or_lpc == "t_env":
		generate_tenv_formant_analysis(audio_file, analysis = analysis_name, nb_formants = nb_formants, wait = True, ana_winsize = ana_winsize)
	
	else:
		print("t_env_or_lpc should be either t_env or lpc")

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
		formants_df = pd.concat([formants_df, formant])

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
	import os
	
	#Take the absolulte path
	audio_file = os.path.abspath(audio_file)
	script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ts_formants.praat')
	_, out = parselmouth.praat.run_file(script, audio_file, str(time_step) , str(nb_formants), str(max_formant_freq), str(window_size), str(pre_emph), capture_output=True)

	content = out.splitlines()

	#parse and organize praat's output
	freqs_df = pd.DataFrame()
	bws_df = pd.DataFrame()
	for line in content:
		line = [ np.nan if x == u'--undefined--' else float(x) for x in line.split()]
		
		frequency = {}
		bandwidth = {}
		for index in range(0,nb_formants):
			frequency['F'+str(index+1)] = line[index*2+1]
			bandwidth['F'+str(index+1)] = line[index*2+2]
		
		freq_df = pd.DataFrame(frequency, index=[line[0]])
		bw_df   = pd.DataFrame(bandwidth, index=[line[0]])
		
		freqs_df = pd.concat([freqs_df, freq_df])
		bws_df   = pd.concat([bws_df, bw_df])

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
		
		#Could use praat here to do this
		harm_time, harm_vals = get_harmonicity_ts(audio_file, time_step=time_step, normalise=True)

		#Add harmonicity to bandwidth
		bws_df = bws_df.reset_index()
		bws_df["harmonicity"] = bws_df.apply(add_harm, axis=1, args=(harm_time, harm_vals,))
		bws_df = bws_df.set_index("time")

		#Add harmonicity to frequency
		freqs_df = freqs_df.reset_index()
		freqs_df["harmonicity"] = freqs_df.apply(add_harm, axis=1, args=(harm_time, harm_vals,))
		freqs_df = freqs_df.set_index("time")

	#Return frequencies and bandwidths dataframes	
	return freqs_df, bws_df

def get_mean_formant_praat_harmonicity(audio_file, time_step=0.001, window_size=0.1, nb_formants=5, max_formant_freq=5500, pre_emph=50.0, harmonicity_threshold=None, formant_method='mean'):
	"""
		get mean formant with praat, but exclucding all values under a certain harmonicity threshold 
		(if harmonicity_threshold is defined, if None, just compute mean of time series)
		It's not recomended to use this function without the harmonicity threshold.
		To keep only harmonic sounds use a threshold > 0.85
		
		Input:
		time_step: The time step to estimate the formants (this is passed to Praat)
		window_size: The window size of the formant analysis IN SECONDS  (this is passed to Praat)
		nb_formants: Number of formants to estimate  (this is passed to Praat)
		max_formant_freq: Maximum formant frequency available, this is an important praat parameter 
						  (5500 usually works well for female voices)
						   Check praat's documentation to know more
		pre_emph: pre emphasis: check praat documentation on formant estimation
		harmonicity_threshold: harmonicity threshold to exclude all the values below a certain harmonic threshold
		formant_method: the method to use to mean across formants. Either 'mean' or 'median' .

		Output;
			Dataframe with mean formant frequencies and bandwidths

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
		
	if formant_method=='median':
		mean_bws_df 	   = bws_df.reset_index().groupby(["Formant"]).median().reset_index()
		bws_std_df 	       = bws_df.reset_index().groupby(["Formant"]).std()
		bws_std_df.drop(["time"], axis=1, inplace=True)
		bws_std_df.columns = [x+"_std" for x in bws_std_df.columns]
		bws_std_df.reset_index(inplace=True)
		bws_df             = mean_bws_df.merge(bws_std_df, on = ["Formant"])
		
		#frequencies
		mean_freqs_df        = freqs_df.reset_index().groupby(["Formant"]).median().reset_index()
		freqs_std_df 	     = freqs_df.reset_index().groupby(["Formant"]).std()
		freqs_std_df.drop(["time"], axis=1, inplace=True)
		freqs_std_df.columns = [x+"_std" for x in freqs_std_df.columns]
		freqs_std_df.reset_index(inplace=True)
		freqs_df = mean_freqs_df.merge(freqs_std_df, on = ["Formant"])

	elif formant_method=='mean':
		mean_bws_df 	   = bws_df.reset_index().groupby(["Formant"]).mean().reset_index()
		bws_std_df 	       = bws_df.reset_index().groupby(["Formant"]).std()
		bws_std_df.drop(["time"], axis=1, inplace=True)
		bws_std_df.columns = [x+"_std" for x in bws_std_df.columns]
		bws_std_df.reset_index(inplace=True)
		bws_df             = mean_bws_df.merge(bws_std_df, on = ["Formant"])
		
		#frequencies
		mean_freqs_df        = freqs_df.reset_index().groupby(["Formant"]).mean().reset_index()
		freqs_std_df 	     = freqs_df.reset_index().groupby(["Formant"]).std()
		freqs_std_df.drop(["time"], axis=1, inplace=True)
		freqs_std_df.columns = [x+"_std" for x in freqs_std_df.columns]
		freqs_std_df.reset_index(inplace=True)
		freqs_df = mean_freqs_df.merge(freqs_std_df, on = ["Formant"])
	

	else:
		raise ValueError("formant_method should be either 'mean' or 'median'")

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

	script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'formants_mean.praat')
	_, x = parselmouth.praat.run_file(script, Fname, capture_output=True)	
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

def get_formant_dispersion(Fname, nb_formants_fd=5
							, time_step=0.001
							, window_size=0.1
							, nb_formants=5
							, max_formant_freq=5500
							, pre_emph=50.0
							, harmonicity_threshold=None
							, formant_method='mean'

	):
	"""
	Inputs:
		Fname 				  : name of the file to use
		nb_formants_fd  : nuùmber of formants to use in the computation of formant dispersion

		harmonicity_threshold : harmonicty threshold to exclude parts of the sound whenc omputing mean formant frequency 
								(between 0 and 1 : 1 is very harmonic, 0 is unharmonic)

		
		nb_formants 	: number of formants to estimate

		Check the help from get_mean_formant_praat_harmonicity to see other detailed parameters

	Output:
		Formant dispersion
	
	Compute and return formant dispersion Following Fitch 1997 JASA Formula

	"""
	
	#Get formant frequencies
	df, db = get_mean_formant_praat_harmonicity(Fname
							, time_step		= time_step
							, window_size	= window_size
							, nb_formants 	= nb_formants
							, max_formant_freq  = max_formant_freq
							, pre_emph 			= pre_emph
							, harmonicity_threshold = harmonicity_threshold
							, formant_method 		= formant_method
		)

	formants = df["Frequency"].values

	#compute formant dispersion
	fd = 0
	for i in range(1, nb_formants_fd):
		fd = fd + formants[i] - formants[i-1]
	fd = fd / (nb_formants_fd -1)

	#return formant dispersion
	return fd

def estimate_vtl(Fname 		, speed_of_sound 		= 335 
							, time_step 			= 0.001
							, window_size 			= 0.1
							, nb_formants 			= 5
							, max_formant_freq 		= 5500
							, pre_emph 				= 50.0
							, harmonicity_threshold = None
							, formant_method 		= 'mean'
				):
	"""
	Estimate vocal tract length following the formula in Fitch 1997
	The speed_of_sound is usually ~335 m/s, and L is the vocal tract length in m.
	
	Inputs:
		Fname 				  : name of the audio file to analyse (only works with mono files)
		harmonicity_threshold : harmonicty threshold to exclude parts of the sound whenc omputing mean formant frequency 
								(between 0 and 1 : 1 is very harmonic, 0 is unharmonic)

		speed_of_sound 		  : speed of sound in m.s-1 (in this case the output will be in meters)
		
		Check the help from get_mean_formant_praat_harmonicity for other detailed inputs

	Outputs:
		vtl : estimated vocal tract length in meters (depends on the unit of speed of sound)

	"""
	#compute formant dispersion
	fd = get_formant_dispersion(Fname 				= Fname
							, time_step 			= time_step
							, window_size 			= window_size
							, nb_formants 			= nb_formants
							, max_formant_freq 		= max_formant_freq
							, pre_emph 				= pre_emph
							, harmonicity_threshold = harmonicity_threshold
							, formant_method 		= formant_method
							)

	#estimate vocal tract length
	vtl = speed_of_sound/(2*fd)

	return vtl

# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ----------------- Spectral Enveloppes
# ----------------- LPC and True Envelope
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#

# ---------- get_mean_lpc_from_audio
def get_mean_lpc_from_audio(audio, wait = True, nb_coefs = 45, destroy_sdif_after_analysis = True):
	"""
	parameters:
		audio 		: file to anlayse
		analysis 	: sdif file to generate, if not defined analysis will have the same name as the audio file but with an sdif extension.
		nb_coefs 	: number of lpc coeficients for the analysis

	warning : 
		this function only works for mono files
	"""
	from parse_sdif import mean_matrix
	path_and_name =  get_file_without_path(audio)
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
			file_tag = get_file_without_path(audio_file)
			analysis = file_tag + ".sdif" 
		else:
			file_tag = get_file_without_path(audio_file)
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
			file_tag = get_file_without_path(audio_file)
			analysis = file_tag + ".sdif" 
		else:
			file_tag = get_file_without_path(audio_file)
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
	from scipy import signal
	import matplotlib.pyplot as plt

	#Read audio file
	sound_in, fs = soundfile.read(file)
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
def get_f0(audio_file, analysis ="", f_min =80, f_max=1500, F=3000, wait = True, destroy_sdif_after_analysis = True):
	"""
    Get f0
    	audio_file : file to analyse

    return : 
    	f0times: time tags
    	f0harm: signal harmonicity
    	f0val: fundamental frequency estimation in Hz

    """
	if os.path.dirname(audio_file) == "":
		file_tag = get_file_without_path(audio_file)
		f0_analysis = file_tag + "f0.sdif" 
		if analysis == "":
			analysis = file_tag + ".sdif" 
	else:
		file_tag = get_file_without_path(audio_file)
		f0_analysis = os.path.dirname(audio_file)+ "/" + file_tag + "f0.sdif"
	
	from super_vp_commands import generate_f0_analysis
	generate_f0_analysis( audio_file, f0_analysis, f_min =f_min, f_max=f_max, F=F, wait = wait)

	
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

	#Read audio file
	data, fs = soundfile.read(audio_file)

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
def get_sound_duration(audio_file):
	"""
	returns sound duration in seconds
	"""
	#Read audio file
	sound_in, sr = soundfile.read(audio_file)

	return len(sound_in)/float(sr)

def get_sf(audio_file):
	"""
	returns sampling frequency
	"""
	#Read audio file
	sound_in, sf = soundfile.read(audio_file)

	return float(sf)

def get_nb_channels(audio_file):
	"""
	get number of channels of audiofile
	"""
	sound_in, sr = soundfile.read(audio_file)
	if len(sound_in.shape) < 2:
		nb_channels= 1
	else:
		nb_channels = sound_in.shape[1]
	
	return nb_channels


def find_silence(audio_file, threshold = -65, wnd_size = 16384):
	"""
	find a segment of silence (<threshold dB)in the sound file
	return tag in seconds
	"""	
	#Read audio file
	x, fs = soundfile.read(audio_file)

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


def get_mean_rms_level_when_sound(source, rms_threshold = -50, window_size = 16384):
	"""
	input:
		source : fsource audio file
		rms_threshold : this is the threshold
		WndSize : window size to compue the RMS on
		target_folder : folder to save the extracted sounds in

	This function separates all the sentences inside an audiofile.
	It takes each sentence and put it into one audio file inside target_folder with the name target_nb
	The default parameters were tested with notmal speech.
	Only works if file is at least 500 ms long, which can be tuned
	You can change the rms threshold to tune the algorithm
	"""
	#Read audio file
	x, fs = soundfile.read(source)

	intervals = get_intervals_of_sound(source, RMS_threshold = rms_threshold, window_size = window_size)
	t, rmss = get_RMS_over_time(source, window_size = window_size)

	median_level=None
	sentences_levels = []
	for interval in intervals:
		begin = interval[0]
		end   = interval[1]    
		
		#get corresponding indexes in t to search in the rmss
		begin = bisect(t, begin)
		end   = bisect(t, end)

		sentences_levels.append(np.mean(rmss[begin:end]))

		median_level = np.nanmedian(sentences_levels)

	if median_level:
		return median_level
	else:
		return np.nan

def get_talking_ts(file, threshold=-50, time_step=0.01):
    silence_tags = find_silence(file, threshold=threshold)
    duration = get_sound_duration(file)
    times = np.arange(0, duration, time_step)

    talking_vals = []
    for time in times:
        talking=True
        for interval in silence_tags:
            if interval[0] < time and time < interval[1]:
                talking=False
                continue
    
        talking_vals.append(talking)
    
    return times, talking_vals

def get_silence_level(source, window_size = 16384):
	"""
	input:
		source : fsource audio file
		WndSize : window size to compue the RMS on

	Return the minimum level in an audio file
	"""
	#Read audio file
	t, rmss = get_RMS_over_time(source, window_size = window_size)

	return np.min(rmss)


# -------------------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------------------#
# ------ VAD - Example code taken from : https://github.com/wiseman/py-webrtcvad/blob/master/example.py  ------#
# -------------------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------------------#
class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    import contextlib
    import wave

    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    import wave

    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n           

def vad_collector(sample_rate, frame_duration_ms,padding_duration_ms, vad, frames):
	"""Filters out non-voiced audio frames.
	Given a webrtcvad.Vad and a source of audio frames, yields only
	the voiced audio. Uses a padded, sliding window algorithm over the audio frames.
	When more than 90% of the frames in the window are voiced (as
	reported by the VAD), the collector triggers and begins yielding
	audio frames. Then the collector waits until 90% of the frames in
	the window are unvoiced to detrigger.
	The window is padded at the front and back to provide a small
	amount of silence or the beginnings/endings of speech around the
	voiced frames.

	Arguments:
	sample_rate : The audio sample rate, in Hz.
	frame_duration_ms : The frame duration in milliseconds.
	padding_duration_ms : The amount to pad the window, in milliseconds.
	vad : An instance of webrtcvad.Vad.
	frames : a source of audio frames (sequence or generator).
	
	Returns : A generator that yields PCM audio data.
	"""
	num_padding_frames = int(padding_duration_ms / frame_duration_ms)

	# We use a deque for our sliding window/ring buffer.
	ring_buffer = collections.deque(maxlen=num_padding_frames)

	# We have two states: TRIGGERED and NOTTRIGGERED. We start in the
	# NOTTRIGGERED state.
	triggered = False

	#To store sentences
	time_tags = []
	sentences = []

	voiced_frames = []
	for frame in frames:
		is_speech = vad.is_speech(frame.bytes, sample_rate)

		if not triggered:
			ring_buffer.append((frame, is_speech))
			num_voiced = len([f for f, speech in ring_buffer if speech])
			
			# If we're NOTTRIGGERED and more than 90% of the frames in
			# the ring buffer are voiced frames, then enter the
			# TRIGGERED state.
			if num_voiced > 0.9 * ring_buffer.maxlen:
				triggered = True
				beg = frame.timestamp
				
				# We want to yield all the audio we see from now until
				# we are NOTTRIGGERED, but we have to start with the
				# audio that's already in the ring buffer.
				for f, s in ring_buffer:
					voiced_frames.append(f)
				ring_buffer.clear()
		else:
			
			# We're in the TRIGGERED state, so collect the audio data
			# and add it to the ring buffer.
			voiced_frames.append(frame)
			ring_buffer.append((frame, is_speech))
			num_unvoiced = len([f for f, speech in ring_buffer if not speech])

			# If more than 90% of the frames in the ring buffer are
			# unvoiced, then enter NOTTRIGGERED and yield whatever
			# audio we've collected.
			if num_unvoiced > 0.8 * ring_buffer.maxlen:
				triggered = False
				end = frame.timestamp

				#Store data
				sentences.append(b''.join([f.bytes for f in voiced_frames]))
				time_tags.append([beg, end])


				ring_buffer.clear()
				voiced_frames = []


	# If we have any leftover voiced audio when we run out of input,
	# yield it.
	if voiced_frames:
		
		sentences.append( b''.join([f.bytes for f in voiced_frames]))
		beg = voiced_frames[0].timestamp
		end = voiced_frames[-1].timestamp
		time_tags.append([beg, end])

	return time_tags, sentences


def get_voiced_segments_vad(audio_file, vad=3, frame_duration_ms=30, padding_duration_ms=300, write_sentences= False, destination_folder="", add_time_tags=False):
	#A frame must be either 10, 20, or 30 ms in duration:
	
	#audio, sample_rate = read_wave("tests/new_sr.wav")
	import webrtcvad
	x, fs = read_wave(audio_file)

	vad = webrtcvad.Vad(vad)
	frames = frame_generator(frame_duration_ms, x, fs)
	frames = list(frames)
	time_tags, sentences = vad_collector(fs, frame_duration_ms=frame_duration_ms, padding_duration_ms=padding_duration_ms, vad=vad, frames=frames)

	#sentences = get_speech_sentences(times, is_speech)

	# if len(sentences) > 0 and write_sentences:
	# 	for index, sentence in enumerate(sentences):
	# 		beg = times.index(sentence[0])
	# 		end = times.index(sentence[1])

	# 		write_wave(destination_folder + str(index)+".wav", x[beg:end], fs)

	if write_sentences:
		for i, sentence in enumerate(sentences):
			if add_time_tags:
				start = time_tags[i][0]
				stop = time_tags[i][1]
				path = destination_folder+"%002d_%00.2f_%00.2f.wav" % (i,start, stop,)
			else:
				path = destination_folder+"%002d.wav" % (i,)

			write_wave(path, sentence, fs)


	return time_tags


# -------------------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------------------#
# ------ AUDIO BATCH Analyses  --------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------------------#

## analyse_audio_file
def analyse_audio_file(file
						, speed_of_sound		= 335
						, time_step				= 0.001
						, window_size			= 0.01 
						, pitch_floor			= 75
						, pitch_ceiling			= 350
						, nb_formants			= 6
						, nb_formants_fd		= 5
						, max_formant_freq		= 5500
						, pre_emph				= 50.0
						, harmonicity_threshold = None 
						, sc_rms_thresh         = -60
						, sc_ws                 = 512
						, parameter_tag         = None
						, verbose = False
						, formant_method		 = 'mean'
						, target_folder			 = "audio_analysis/"
						):
	"""
	This functions extracts mean sound features of a specific recording.
	Harmonicity threshold : is used to clean formant and pitch values
	sc_rms_thresh : RMS threshold used to clean SC values

	"""
	#handle name of file
	base = get_file_without_path(file)
	
	#File to save analyses
	analysis_file = target_folder + base + ".csv"

	if not os.path.isfile(analysis_file):
		open(analysis_file, 'a').close()
	else:
		print("Analysis for file "+ analysis_file +" exists, skipping ")
	

	if verbose:
		print("Starting analysis for file : " + file)

	#Handle updated parameter tags
	if parameter_tag:
		index = parameter_tag[0]
		key   = base[index]
		parameters = parameter_tag[1][key]

		#update analysis parameters to match the file tag
		if 'harmonicity_threshold' in list(parameters.keys()): harmonicity_threshold = parameters['harmonicity_threshold']
		if 'time_step' 			   in list(parameters.keys()): time_step             = parameters['time_step'] 	
		if 'window_size' 		   in list(parameters.keys()): window_size           = parameters['window_size'] 	
		if 'nb_formants' 		   in list(parameters.keys()): nb_formants           = parameters['nb_formants'] 		   
		if 'nb_formants_fd' 	   in list(parameters.keys()): nb_formants_fd        = parameters['nb_formants_fd'] 	   
		if 'max_formant_freq' 	   in list(parameters.keys()): max_formant_freq      = parameters['max_formant_freq'] 	   
		if 'pre_emph' 			   in list(parameters.keys()): pre_emph              = parameters['pre_emph'] 	
		if 'pitch_floor' 		   in list(parameters.keys()): pitch_floor           = parameters['pitch_floor'] 
		if 'pitch_ceiling' 		   in list(parameters.keys()): pitch_ceiling         = parameters['pitch_ceiling'] 			   				   


	#Get sound duration
	sound_duration = get_sound_duration(file)
	if sound_duration < time_step:
		print("not analysing " + file+ " because it's shorter than the time step")
		return

	#Get formant frequencies
	df, db = get_mean_formant_praat_harmonicity(file 
										, time_step        = time_step
										, window_size      = window_size
										, nb_formants      = nb_formants
										, max_formant_freq = max_formant_freq
										, pre_emph         = pre_emph
										, harmonicity_threshold   = harmonicity_threshold
										, formant_method		  = formant_method
										)

	#Formants
	formants     = df["Frequency"].values
	formants_std = df["Frequency_std"].values

	#Bandwidths
	bws     = db["Bandwidth"].values
	bws_std = db["Bandwidth_std"].values


	#compute formant dispersion & vtl
	if len(formants) >= nb_formants_fd:
		fd = 0
		for i in range(1, nb_formants_fd):
			fd = fd + formants[i] - formants[i-1]
		fd = fd / (nb_formants_fd -1)

		#Compute vocal tract length following Fitch 1997 formulae
		vtl = speed_of_sound/(2*fd)

	else:
		fd  = float("nan")
		vtl = float("nan")

	#pitch descriptors
	times, f0s = Extract_ts_of_pitch_praat(file, time_step=time_step , pitch_floor = pitch_floor, pitch_ceiling =  pitch_ceiling, harmonicity_threshold=harmonicity_threshold)

	pitch_mean = np.nanmean(f0s)
	pitch_std = np.nanstd(f0s)

	#compute mean centroid
	centroid = get_mean_spectral_centroid_when_sound(file, RMS_threshold = sc_rms_thresh, window_size = sc_ws)

	#RMS related features : 
	#silence_level            = get_silence_level(file, window_size_silence_level=window_size_silence_level)
	#mean_rms                 = get_mean_rms_level_when_sound(rms_threshold= rms_threshold, window_size= window_size_rms)
	#mean_rms_when_sound      = get_mean_rms_level_when_sound(rms_threshold= rms_threshold, window_size= window_size_rms)
	#TODO : Add time talking

	#create dictionary with values and then append to general dataframe
	aux_df = pd.DataFrame()

	#Add fle and CGS
	aux_df["File"] = [base]
	aux_df["CGS"] = [centroid]
	
	#Add formant information
	if len(formants) == nb_formants:
		#Add formants
		for index in range(0, nb_formants):
			aux_df["F"+str(index+1)] = [formants[index]]
			aux_df["F"+str(index+1)+"_std"] = [formants_std[index]]

			#Add Bandwidths
			aux_df["F"+str(index+1)+"bw"] = [bws[index]]
			aux_df["F"+str(index+1)+"bw_std"] = [bws_std[index]]
	else:
		#Add nans everywhere
		for index in range(0, nb_formants):
			aux_df["F"+str(index+1)] = [float("nan")]
			aux_df["F"+str(index+1)+"_std"] = [float("nan")]
		
			#Add Bandwidths
			aux_df["F"+str(index+1)+"bw"] = [float("nan")]
			aux_df["F"+str(index+1)+"bw_std"] = [float("nan")]


	#Add other descriptors
	aux_df["pitch_mean"]   	= [pitch_mean]
	aux_df["pitch_std"]   	= [pitch_std]
	aux_df["formant_disp"] 	= [fd]
	aux_df["vtl"]   		= [vtl]
	aux_df["sound_duration"]  = [sound_duration]

	#append df to general dataframe
	aux_df.to_csv(analysis_file)
	
## analyse_audio_folder
def analyse_audio_folder(source_folder
						, speed_of_sound		= 335
						, time_step				= 0.001
						, window_size			= 0.01 
						, pitch_floor			= 75
						, pitch_ceiling			= 350
						, nb_formants			= 6
						, nb_formants_fd		= 5
						, max_formant_freq		= 5500
						, pre_emph				= 50.0
						, harmonicity_threshold = None 
						, sc_rms_thresh         = -60
						, sc_ws                 = 512
						, parameter_tag         = None
						, verbose = False
						, formant_method		 = 'mean'
						, target_folder			 = "audio_analysis/"
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
									For example (0, {F: {'max_formant_freq':5500} , M:{'max_formant_freq':4900} })

		verbose: for debuging purposes, put this to True to print each analysed file
		formant_method   	  : the method to use to colapse formants. Either 'mean' or 'median'.

	Output:
		Beautiful dataframe with all the features

	"""
	try:
		os.mkdir(target_folder)
	except:
		pass
	
	#for each file in source folder, get all the features
	for file in glob.glob(source_folder):
		analyse_audio_file(file
					    , speed_of_sound		= speed_of_sound
						, time_step				= time_step
						, window_size			= window_size
						, pitch_floor			= pitch_floor
						, pitch_ceiling			= pitch_ceiling
						, nb_formants			= nb_formants
						, nb_formants_fd		= nb_formants_fd
						, max_formant_freq		= max_formant_freq
						, pre_emph				= pre_emph
						, harmonicity_threshold = harmonicity_threshold 
						, sc_rms_thresh         = sc_rms_thresh
						, sc_ws                 = sc_ws
						, parameter_tag         = parameter_tag
						, verbose = verbose
						, formant_method		 = formant_method
						, target_folder			 = target_folder
						)

## analyse_audio_folder_parallel
def analyse_audio_folder_parallel(source_folder
						, speed_of_sound		= 335
						, time_step				= 0.001
						, window_size			= 0.01 
						, pitch_floor			= 75
						, pitch_ceiling			= 350
						, nb_formants			= 6
						, nb_formants_fd		= 5
						, max_formant_freq		= 5500
						, pre_emph				= 50.0
						, harmonicity_threshold = None 
						, sc_rms_thresh         = -60
						, sc_ws                 = 512
						, parameter_tag         = None
						, verbose               = False
						, formant_method		 = 'mean'
						, target_folder			 = "audio_analysis/"
						):
	"""
	Analyse all the sounds from a folder and returns a data frame with key vocal features using parallel computations (In different CPU cores).
	Inputs:
		source_folder 		  : Folder with all the sounds to analyse, prefer mono files, and wav
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
		harmonicity_threshold : Harmonicity Threshold to filter fundamental and formant frequencies
		sc_rms_thresh		  : Amplitude threshold to remove from sound in spectral centroid analysis (in dB)
		sc_ws 		  		  : Window Size in samples for the spectral centoid analysis
		parameter_tag		  : parameter tag should be composed of two values.
									First, the value of the position in the name file where the tag is
									Second, a dictionary with a dictionary with the corresponding parameters to use for each tag
									For example (0, {F: {'max_formant_freq':5500} , M:{'max_formant_freq':4900} })

		verbose: for debuging purposes, put this to True to print each analysed file
		formant_method   	  : the method to use to colapse formants. Either 'mean' or 'median'.

	Output:
		Beautiful dataframe with all the features

	"""
	import multiprocessing
	from itertools import repeat	

	try:
		os.mkdir(target_folder)
	except:
		pass

	pool_obj = multiprocessing.Pool()
	files     = glob.glob(source_folder)

	pool_obj.starmap(analyse_audio_file, zip(files
										, repeat(speed_of_sound)
										, repeat(time_step)
										, repeat(window_size)
										, repeat(pitch_floor)
										, repeat(pitch_ceiling)
										, repeat(nb_formants)
										, repeat(nb_formants_fd)
										, repeat(max_formant_freq)
										, repeat(pre_emph)
										, repeat(harmonicity_threshold)
										, repeat(sc_rms_thresh)
										, repeat(sc_ws)
										, repeat(parameter_tag)
										, repeat(verbose)
										, repeat(formant_method)
										, repeat(target_folder)
										)
					)

## analyse_audio_file_ts
def analyse_audio_file_ts(file
						, time_step				= 0.001
						, praat_ws			    = 0.01 
						, sc_ws                 = 512
						, rms_ws                = 512
						, pitch_floor			= 75
						, pitch_ceiling			= 350
						, nb_formants			= 6
						, max_formant_freq		= 5500
						, pre_emph				= 50.0
						, parameter_tag         = None
						, verbose               = False
						, silence_threshold      = -50
						, target_folder			 = "audio_analysis/"
						, plot_features          = False
						):
	"""
	This functions extracts mean sound features of a specific recording.
		sc_ws = this is in samples, the size of the FFT
	"""
	#handle name of file
	file_tag = get_file_without_path(file)

	#Create result folder
	os.makedirs(target_folder, exist_ok=True)
	
	#File to save analyses
	analysis_file = target_folder + file_tag + ".csv"

	if not os.path.isfile(analysis_file):
		open(analysis_file, 'a').close()
	else:
		print("Analysis for file "+ analysis_file +" exists, skipping ")
		return

	if verbose:
		print("Starting analysis for file : " + file, flush=True)

	#Handle updated parameter tags
	if parameter_tag:
		index = parameter_tag[0]
		key   = file_tag[index]
		parameters = parameter_tag[1][key]

		#update analysis parameters to match the file tag
		if 'harmonicity_threshold' in list(parameters.keys()): harmonicity_threshold = parameters['harmonicity_threshold']
		if 'time_step' 			   in list(parameters.keys()): time_step             = parameters['time_step'] 	
		if 'window_size' 		   in list(parameters.keys()): window_size           = parameters['window_size'] 	
		if 'nb_formants' 		   in list(parameters.keys()): nb_formants           = parameters['nb_formants'] 		   
		if 'nb_formants_fd' 	   in list(parameters.keys()): nb_formants_fd        = parameters['nb_formants_fd'] 	   
		if 'max_formant_freq' 	   in list(parameters.keys()): max_formant_freq      = parameters['max_formant_freq'] 	   
		if 'pre_emph' 			   in list(parameters.keys()): pre_emph              = parameters['pre_emph'] 
		if 'pitch_floor' 		   in list(parameters.keys()): pitch_floor           = parameters['pitch_floor'] 
		if 'pitch_ceiling' 		   in list(parameters.keys()): pitch_ceiling         = parameters['pitch_ceiling'] 			   



	#Get formant frequencies
	if verbose:
		print("start formant analysis", flush=True)
	freqs_df, bws_df =	get_formant_ts_praat(file
											, time_step=time_step
											, window_size=praat_ws
											, nb_formants=nb_formants
											, max_formant_freq=max_formant_freq
											, pre_emph=pre_emph
											, add_harmonicity=False
											)
	freqs_df = freqs_df.reset_index().pivot(index=["time"], columns="Formant",values="Frequency").rename_axis(None, axis=1)
	bws_df   = bws_df.reset_index().pivot(index=["time"], columns="Formant",values="Bandwidth").rename_axis(None, axis=1)

	#pitch descriptors
	if verbose:
		print("start fundamental freq analysis", flush=True)
	t_f0, f0s = Extract_ts_of_pitch_praat(file, time_step=time_step , pitch_floor = pitch_floor, pitch_ceiling =  pitch_ceiling)

	#compute mean centroid
	if verbose:
		print("Start spectral centroid", flush=True)
	t_cent, centroids = get_spectral_centroid(file, window_size = sc_ws, noverlap = 0, plot_specgram = False)

	#RMS
	if verbose:
		print("Start RMS analysis", flush=True)
	t_rms, rmss = get_RMS_over_time(file, window_size = rms_ws)

	# Create time series talking vs not talking	
	if verbose:
		print("Start talking TS", flush=True)
	t_talking, talking_vals =get_talking_ts(file, threshold=silence_threshold, time_step=time_step)

	#Harmonicity
	if verbose:
		print("Start harmonicity", flush=True)
	t_harm, harms = get_harmonicity_ts(file, time_step)

	duration = get_sound_duration(file)
	time_vals = np.arange(0, duration, time_step)

	df = pd.DataFrame(index=time_vals)

	for formant in range(1, nb_formants+1):
		df["F"+str(formant)+"_freq"] = np.interp(time_vals, freqs_df.index.values, freqs_df["F"+str(formant)])
		df["F"+str(formant)+"_bw"]   = np.interp(time_vals, bws_df.index.values, bws_df["F"+str(formant)])

	df["f0"]          = np.interp(time_vals, t_f0, f0s)
	df["centroid"]    = np.interp(time_vals, t_cent, centroids)
	df["rms"]         = np.interp(time_vals, t_rms, rmss)
	df["talking"]     = np.interp(time_vals, t_talking, talking_vals)
	df["harmonicity"] = np.interp(time_vals, t_harm, harms)

	#Other info
	df["file_tag"] = [file_tag for i in range(len(df))]

	#Save analyses to dataframe
	if verbose:
		print("--- Saving results ---", flush=True)
		print(df, flush=True)
	df.to_csv(analysis_file)

	#Plot features
	if plot_features:
		import matplotlib.pyplot as plt

		fig, axes = plt.subplots(nrows=6, figsize=(23,15), sharex=True)

		for formant in range(1, nb_formants+1):
			axes[0].plot(df["F"+str(formant)+"_freq"].values, label="F"+str(formant))

		axes[1].plot(df["f0"].values, label='f0s')
		axes[2].plot(df["rms"].values, label='rms')
		axes[3].plot(df["talking"].values, label='talking')
		axes[4].plot(df["harmonicity"].values, label='harmonicity')
		axes[5].plot(df["centroid"].values, label='centroid')

		# Add titles to each subplot
		axes[0].set_title('Formants')
		axes[1].set_title('f0s')
		axes[2].set_title('rms')
		axes[3].set_title('talking')
		axes[4].set_title('harmonicity')
		axes[5].set_title('centroid')

		# Add labels and legend to the last subplot
		axes[-1].set_xlabel('Time')
		axes[-1].set_ylabel('Values')
		axes[-1].legend()

		# Adjust layout
		plt.tight_layout()

		#Save plot
		plt.savefig(target_folder + file_tag + ".pdf")

# analyse_audio_file_ts_folder_parallel
def analyse_audio_ts_folder(source_folder  
							, time_step				= 0.001
							, praat_ws			    = 0.01 
							, sc_ws                 = 512
							, rms_ws                = 512
							, pitch_floor			= 75
							, pitch_ceiling			= 350
							, nb_formants			= 6
							, max_formant_freq		= 5500
							, pre_emph				= 50.0
							, parameter_tag         = None
							, verbose = False
							, silence_threshold      = -50
							, target_folder			 = "audio_analysis/"
							, plot_features          = True
							):
	"""
	Extract Time series of features in parallel CPU cores
	Check analyse_audio_file_ts for details
	"""
	import multiprocessing
	from itertools import repeat	
	#Create target forlder recursively if needed
	os.makedirs(target_folder, exist_ok=True)

	print("Start of analyse_audio_ts_folder", flush=True)
	pool_obj = multiprocessing.Pool()
	files     = glob.glob(source_folder)
	pool_obj.starmap(analyse_audio_file_ts, zip(files
										, repeat(time_step)				
										, repeat(praat_ws)			    
										, repeat(sc_ws)                
										, repeat(rms_ws)          
										, repeat(pitch_floor)		
										, repeat(pitch_ceiling)			
										, repeat(nb_formants)	
										, repeat(max_formant_freq)		
										, repeat(pre_emph)		
										, repeat(parameter_tag)
										, repeat(verbose) 
										, repeat(silence_threshold)      
										, repeat(target_folder)
										, repeat(plot_features)
										)
					)