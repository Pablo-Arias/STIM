#-*- coding: utf-8 -*-
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ---------- Made by pablo Arias @ircam on 11/2015
# ---------- Copyright (c) 2018 CREAM Lab // CNRS / IRCAM / Sorbonne Universite
# ----------
# ---------- execute different super vp commands
# ---------- to use this don't forget to include these lines before your script:
# ----------
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#

from __future__ import absolute_import
from __future__ import print_function
import shlex, subprocess
import os
import numpy as np
from super_vp_path import get_super_vp_path
super_vp_path = get_super_vp_path()
import random
import string

# ---------- generate_f0_analysis
def generate_f0_analysis(audio_file, analysis ="", f_min =80, f_max=1500, F=3000, wait = True):
	"""
	file : file to anlayse
	analysis : sdif file to generate, if not defined analysis will be saved in the same path of the audio file, with the same name but with an sdif extension.
	f_min : Minimum frequency value for pitch analysis in Hz
	f_max : Maximum frequency value for pitch analysis in Hz
	F     : Maximum frequency in spectrum in Hz
	wait : wait until the analysis is finished before exit
	"""
	if analysis =="":
		file_tag = os.path.basename(audio_file)
		file_tag = os.path.splitext(file_tag)[0]
		analysis = os.path.join(os.path.dirname(audio_file), file_tag + ".sdif")

	# parameters  = "-t -ns  -S"+audio_file+" -Af0 \"fm100, fM1500, F3000, sn120, smooth3 ,Cdem, M0.26 , E0.14\" -Np2 -M0.0464299991726875s -oversamp 8 -Wblackman  -Of4 " + analysis
	
	parameters  = "-t -ns  -S"+audio_file+" -Af0 \"fm"+str(f_min)+", fM"+str(f_max)+", F"+str(F)+", sn120, smooth3 ,Cdem, M0.26 , E0.14\" -Np2 -M0.0464299991726875s -oversamp 8 -Wblackman  -Of4 " + analysis

	cmd 		= super_vp_path + " " + parameters
	args 		= shlex.split(cmd)
	p 			= subprocess.Popen(args)
	if wait:
		p.wait() #wait

	return analysis


# ---------- generate_LPC_analysis
def generate_LPC_analysis(audio_file, analysis="", wait = True, nb_coefs = 45):
	"""
	file : file to anlayse
	analysis : sdif file to generate, if not defined analysis will be saved in the same path of the audio file, with the same name but with an sdif extension.
	"""
	if analysis== "":
		file_tag = os.path.basename(audio_file)
		file_tag = os.path.splitext(file_tag)[0]
		analysis = os.path.dirname(audio_file)+ "/" + file_tag + ".sdif"

	parameters 	=  "-t -ns  -S"+audio_file+" -Alpc "+str(nb_coefs)+"  -Np2 -M0.0907029509544373s -oversamp 16 -Whanning  -OS1  " + analysis
	cmd 		= super_vp_path + " " + parameters
	args 		= shlex.split(cmd)
	p 			= subprocess.Popen(args)	
	if wait:
		p.wait() #wait


# ---------- generate_formant_analysis
def generate_formant_analysis(audio_file, analysis = "", nb_formants = 5, wait = True, ana_winsize = 512):
	"""
	file : file to anlayse
	analysis : sdif file to generate, if not defined analysis will be saved in the same path of the audio file, with the same name but with an sdif extension.
	ana_winsize : Window for the svp formant analysis, in samples

	Be careful, this function works only with mono files
	"""

	nb_formants = str(nb_formants) 
	ana_winsize = str(ana_winsize)

	if analysis =="":
		if os.path.dirname(audio_file) == "":

			file_tag = os.path.basename(audio_file)
			file_tag = os.path.splitext(file_tag)[0]
			analysis = file_tag + ".sdif" 
		else:
			file_tag = os.path.basename(audio_file)
			file_tag = os.path.splitext(file_tag)[0]
			analysis = os.path.dirname(audio_file)+ "/" + file_tag + ".sdif"

	parameters 	=  "-t -ns -S"+audio_file+" -Aformant_lpc n"+nb_formants+" 45  -Np0 -M"+ana_winsize+" -oversamp 8 -Whanning  -OS1 "+ analysis


	cmd 		= super_vp_path + " " + parameters
	args 		= shlex.split(cmd)
	p 			= subprocess.Popen(args)
	if wait:
		p.wait() #wait


# ---------- generate_formant_analysis
def generate_tenv_formant_analysis(audio_file, analysis = "", nb_formants = 5, wait = True, f0="max", ana_winsize = 512):
	"""
	Description:
		Generate an sdif formant analysis using the true envelope 

	file : file to anlayse
	analysis : sdif file to generate, if not defined analysis will be saved in the same path of the audio file, with the same name but with an sdif extension.
	ana_winsize : Window for the svp formant analysis, in samples

	Be careful, this function works only with mono files
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
		if analysis == "":
			analysis = os.path.dirname(audio_file)+ "/" + file_tag + ".sdif"

	generate_f0_analysis( audio_file, f0_analysis)

	if not isinstance(f0, int) and not isinstance(f0, float):
		from parse_sdif import get_f0_info
		f0times, f0harm, f0val = get_f0_info(f0_analysis)
		if f0 == "mean":
			f0 = np.mean(f0val)
		elif f0 == "max":
			f0 = np.max(f0val)
		else:
			print("f0 error! f0 should be either an int, a float or the strings \"mean\" or \"max\"")			

	if analysis =="":
		if os.path.dirname(audio_file) == "":

			file_tag = os.path.basename(audio_file)
			file_tag = os.path.splitext(file_tag)[0]
			analysis = file_tag + ".sdif" 
		else:
			file_tag = os.path.basename(audio_file)
			file_tag = os.path.splitext(file_tag)[0]
			analysis = os.path.dirname(audio_file)+ "/" + file_tag + ".sdif"

	nb_formants = str(nb_formants)
	ana_winsize = str(ana_winsize)

	parameters 	=  "-t -ns -S"+audio_file+" -Atenv +"+str(f0)+"Hz -F0 "+f0_analysis+" -Aformant_tenv n"+nb_formants+" +2,1 -Np0 -M"+ana_winsize+" -oversamp 8 -Whanning  -OS1 "+ analysis
	#parameters 	=  "-t -ns  -S"+audio_file+" -Atenv +"+str(f0)+"Hz -F0 "+f0_analysis+" -Np2 -M0.0907029509544373s -oversamp 16 -Whanning -OS1 "+analysis


	cmd 		= super_vp_path + " " + parameters
	args 		= shlex.split(cmd)
	p 			= subprocess.Popen(args)
	if wait:
		p.wait() #wait


# ---------- generate_true_env_analysis
def generate_true_env_analysis(audio_file, analysis="", wait = True, f0 = "max", oversamp=16):
	"""
	file : file to anlayse
	analysis : sdif file to generate, if not defined analysis will be saved in the same path of the audio file, with the same name but with an sdif extension.
	f0 : Saying the max of f0 is a very important parameter for computing the True Envelope. Those there are diferent options for it, choose it wisely:
		"mean" : Extracts and takes the mean of f0 
		"max"  : Extracts and takes the max of f0
		int or float : you can also specify directly the f0 by giving a float
	
	Be careful, this function works only with mono files
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
		if analysis == "":
			analysis = os.path.dirname(audio_file)+ "/" + file_tag + ".sdif"
	
	generate_f0_analysis( audio_file, f0_analysis)

	if not isinstance(f0, int) and not isinstance(f0, float):
		from parse_sdif import get_f0_info
		f0times, f0harm, f0val = get_f0_info(f0_analysis)
		if f0 == "mean":
			f0 = np.mean(f0val)
		elif f0 == "max":
			f0 = np.max(f0val)
		else:
			print("f0 error! f0 should be either an int, a float or the strings \"mean\" or \"max\"")

	parameters 	=  "-t -ns  -S"+audio_file+" -Atenv +"+str(f0)+"Hz -F0 "+f0_analysis+" -Np2 -M0.0907029509544373s -oversamp "+str(oversamp)+" -Whanning -OS1 "+analysis
	#parameters 	=  "-t -ns  -S"+audio_file+" -Atenv +"+str(f0)+"Hz -F0 "+f0_analysis+" -Np2 -M0.0464399084448814s -oversamp "+str(oversamp)+" -Whanning -OS1 "+analysis	
	#parameters 	=  "-t -ns  -S"+audio_file+" -Atenv "+str(f0)+"Hz -Np2 -M0.0907029509544373s -oversamp 16 -Whanning  -OS0 "+analysis

	cmd 		= super_vp_path + " " + parameters
	args 		= shlex.split(cmd)
	p 			= subprocess.Popen(args)

	if wait:
		p.wait() #wait

	os.remove(f0_analysis)



# ---------- generate_fft_analysis
def generate_fft_analysis(audio_file, analysis = "", wait = True):
	"""
	audio_file : file to anlayse
	analysis : sdif file to generate, if not defined analysis will be saved in the same path of the audio file, with the same name but with an sdif extension.
	"""
	if analysis =="":
		if os.path.dirname(audio_file) == "":

			file_tag = os.path.basename(audio_file)
			file_tag = os.path.splitext(file_tag)[0]
			analysis = file_tag + ".sdif" 
		else:
			file_tag = os.path.basename(audio_file)
			file_tag = os.path.splitext(file_tag)[0]
			analysis = os.path.dirname(audio_file)+ "/" + file_tag + ".sdif"

	parameters  = "-t -ns  -S"+audio_file+" -Afft  -Np4 -M0.0454544983804226s -oversamp 8 -Whanning  -OS1  "+ analysis
	cmd 		= super_vp_path + " " + parameters
	args 		= shlex.split(cmd)
	p 			= subprocess.Popen(args)
	if wait:
		p.wait() #wait

	return analysis	
	


# ---------- transpose_sound
def transpose_sound(source_sound, nb_cents
					, target_sound
					, window_size=512
					, fft_size=512
					, win_oversampling=64
					, fft_oversampling=2
					, print_command_line=False
					, preserve_spectral_envelope=True
					, perserve_transient=True 
					, verbose=False
					, low_pass_filter_freq=None
					, wait = True
					, loudness_norm=False
					):
	"""
	transpose source sound from number of cents and generate target_sound

	source_sound: audio source file
	target_sound: audio file to be created
	nb_cents : number of cents
	
	Check super vp help to see how each of these work more in depth
	"""	

	#Select tags
	#preserve spectral envelope
	if preserve_spectral_envelope:
		transp_type = " -transke " # preserve spectral envelope
	else:
		transp_type = " -trans " # don't preserve

	if loudness_norm:
		loudness_flag = " -envpl 1 "
	else:
		loudness_flag= " "
	
	#verbose baby
	if verbose:
		verbose_flag= " -v "
	else:
		verbose_flag= ""

	#transient preservation infomration
	# With argument "0"  phase synchronization  is switched on.   The argument
	# "1" enables  transient detection  and preservation.  -P  applies to
	# the first track and -p to the second track.
	# Transient preservation localizes transients peaks in time/frequency
	# domain  and re-initializes phases  for the  related bins  after the
	# transient  has  passed  in order  to  keep  the  wave form  of  the
	# transient.
	if perserve_transient:
		transient = " -P1 "
	else:
		transient =" "
		
	if low_pass_filter_freq != None:
		low_pass_tag = " -tr_filt "+str(low_pass_filter_freq) + " "
	else:
		low_pass_tag=""	

	window_size      = str(window_size)
	fft_size         = str(fft_size)
	fft_oversampling = str(int(fft_oversampling))
	win_oversampling = str(win_oversampling)


	parameters =  "-t -Afft "+ fft_size+ " -M"+window_size+" -Np"+fft_oversampling+" -oversamp "+win_oversampling +" -Z"+transp_type+str(nb_cents)+ loudness_flag + verbose_flag+transient+low_pass_tag +"-S"+ source_sound+ " " + target_sound

	cmd 		= super_vp_path + " " + parameters
	args 		= shlex.split(cmd)
	p 			= subprocess.Popen(args)

	if wait:
		p.wait() #wait
	
	if print_command_line:
		print("supervp parameters used are : " + parameters)

# -----------
def dynamic_f0_transposition(times
							, transpositions
							, source_sound
							, target_sound
							, window_size=1024
							, fft_size=512
							, win_oversampling=64
							, fft_oversampling=2
							, print_command_line=False
							, preserve_spectral_envelope=True
							, perserve_transient=True 
							, verbose=False
							, low_pass_filter_freq=None
							, wait = True
							, loudness_norm = False
							):
	"""
		Use super vp to do a dynamic f0 transposition
		times : array with times where to apply the transposition in seconds
		transpositions : array with transposition values in cents
		source_sound : input file
		target_sound : output file to be created
		Other parameters are super vp parameters
	"""
	string_length = 16
	pitch_txt = ''.join(random.choice(string.ascii_lowercase[:6]) for _ in range(string_length))

	with open(pitch_txt, 'w') as file:
		combined_elements = [f'{x} {y}'+"\n" for x, y in zip(times, transpositions)]
		combined_elements = ''.join(combined_elements)
		file.writelines(combined_elements+ "\n")

	transpose_sound(source_sound   = source_sound
				    , nb_cents     = pitch_txt
					, target_sound = target_sound
					, wait         = wait
					, window_size=window_size
					, fft_size=fft_size
					, win_oversampling=win_oversampling
					, fft_oversampling=fft_oversampling
					, print_command_line=print_command_line
					, preserve_spectral_envelope=preserve_spectral_envelope
					, perserve_transient=perserve_transient
					, verbose=verbose
					, low_pass_filter_freq=low_pass_filter_freq
					, loudness_norm = loudness_norm
			)
	
	os.remove(pitch_txt)


# ---------- env_transpose
def env_transpose(source_sound, target_sound, nb_cents, wait = True, env_warp_ws=512, lpc_order=25 , win_oversampling = 64, fft_oversampling=2):
	"""
	transpose source sound from number of cents and generate target_sound

	source_sound: audio source file
	target_sound: audio file to be created
	nb_cents : number of cents
	"""	

	env_warp_ws      = str(env_warp_ws)
	lpc_order        = str(lpc_order)
	fft_oversampling = str(int(fft_oversampling))
	win_oversampling = str(win_oversampling)

	parameters =  "-t -Afft "+lpc_order+" -M"+env_warp_ws+" -Np"+fft_oversampling+" -oversamp "+win_oversampling +" -Z -transenv "+str(nb_cents)+" -S"+ source_sound+ " " + target_sound

	cmd 		= super_vp_path + " " + parameters
	args 		= shlex.split(cmd)
	p 			= subprocess.Popen(args)

	if wait:
		p.wait() #wait

# ---------- freq_warp
def freq_warp(source_sound, target_sound, warp_file, freq_warp_ws=512,lpc_order=25 ,wait = True, warp_method="lpc", win_oversampling = 64, fft_oversampling=2):
	"""
	source_sound     : audio source file
	target_sound     : audio file to be created
	warpfile 	     : file to do the warp.
	freq_warp_ws     : size of the individual signal segments that will be treated by SuperVP
	lpc_order        : order for the lpc analysis
	warp_method      : warp with the lpc or the true envelope
	fft_oversampling : frequency oversampling defined as an exponent that changes the FFT size (FFTsize=freq_warp_ws*2^fft_oversampling)
	win_oversampling : window oversampling or minimum overlap  of analysis and  synthesis windows (hopsize=freq_warp_ws/win_oversampling). Previously called 'freq_warp_oversampling'
	"""

	if os.path.dirname(source_sound) == "":

		file_tag = os.path.basename(source_sound)
		file_tag = os.path.splitext(file_tag)[0]
		f0_analysis = file_tag + "f0.sdif" 
	else:
		file_tag = os.path.basename(source_sound)
		file_tag = os.path.splitext(file_tag)[0]
		f0_analysis = os.path.dirname(source_sound)+ "/" + file_tag + "f0.sdif"

	freq_warp_ws     = str(freq_warp_ws)
	lpc_order        = str(lpc_order)
	fft_oversampling = str(int(fft_oversampling))
	win_oversampling = str(win_oversampling)

	#see super-vp help to understand the parameters
	
	if "lpc" == warp_method:
		# parameters =  "-t -Afft -M512 -N512 -Np2 -oversamp 64 -f0 "+f0_analysis +" -Z -envwarp " + warp_file + " -S" + source_sound + " " + target_sound # warping with manual -N window size and redundant -Np
		# parameters =  "-t -Afft "+lpc_order+" -M"+str(freq_warp_ws)+" -N"+freq_warp_ws+" -Np2 -oversamp "+str(win_oversampling) +" -Z -envwarp " + warp_file + " -S" + source_sound + " " + target_sound # warping before 'win_oversampling' change
		parameters =  "-t -Afft "+lpc_order+" -M"+freq_warp_ws+" -Np"+fft_oversampling+" -oversamp "+win_oversampling +" -Z -envwarp " + warp_file + " -S" + source_sound + " " + target_sound
	
	elif "t_env" == warp_method:
		generate_f0_analysis( source_sound, f0_analysis)
		parameters =  "-t -Afft -atenv -F0 "+f0_analysis+" -M"+freq_warp_ws+" -Np"+fft_oversampling+" -oversamp "+win_oversampling+" -F0 "+f0_analysis +" -Z -envwarp " + warp_file + " -S" + source_sound + " " + target_sound
	else:
		print("wrong warping method specified")
	
	cmd 		= super_vp_path + " " + parameters
	args 		= shlex.split(cmd)
	p 			= subprocess.Popen(args)

	if wait:
		p.wait() #wait
	
	if "t_env" == warp_method:
		os.remove(f0_analysis)


def warp_audio_file(source_freq, target_freq, original_sound, target_sound, freq_warp_ws=512, lpc_order=25, wait=True, warp_method='lpc', win_oversampling=64, fft_oversampling=2):
	"""
	A useful function to use freq warp without writing the warp files.
	freq_warp_ws     : size of the individual signal segments that will be treated by SuperVP
	lpc_order        : order for the lpc analysis
	warp_method      : warp with the lpc or the true envelope
	fft_oversampling : frequency oversampling defined as an exponent that changes the FFT size (FFTsize=freq_warp_ws*2^fft_oversampling)
	win_oversampling : window oversampling or minimum overlap  of analysis and  synthesis windows (hopsize=freq_warp_ws/win_oversampling). Previously called 'freq_warp_oversampling'
	"""
	from audio_analysis import get_sf

	string_length = 16
	warp_txt = ''.join(random.choice(string.ascii_lowercase[:6]) for _ in range(string_length))
	sf = get_sf(original_sound)

	# create warp file
	combined_elements = [f'{x} {y}' for x, y in zip(source_freq, target_freq)]
	combined_elements = ' '.join(combined_elements)
	warp_line = ( str(len(source_freq) + 2) +"\n"
				+ "0.0"  # time
				+ " "
				+ "0.0 0.0 " 
				+ combined_elements
				+ " " + str(sf/2) +" " + str(sf/2)
				)

	with open(warp_txt, 'w') as file:
		file.writelines(warp_line)
		file.writelines("\n")

	freq_warp(source_sound   = original_sound
				, target_sound = target_sound
				, warp_file = warp_txt
				, freq_warp_ws=freq_warp_ws
				, lpc_order=lpc_order
				, wait=wait
				, warp_method=warp_method
				, win_oversampling=win_oversampling
				, fft_oversampling=fft_oversampling
				)

	os.remove(warp_txt)

def stretch_sound_to_target_duration(source_sound, target_sound, target_duration, extreme_precision = True):
	"""
	stretch source sound to target duration

	source_sound: audio source file
	target_sound: audio file to be created
	"""	
	import soundfile
	x, fs = soundfile.read(source_sound)

	source_sound_duration = len(x)/float(fs)
	change_factor = target_duration/source_sound_duration

	#create bpf
	bpf_file = 'aux_file_for_stretch'
	with open(bpf_file,'w+') as new:
		new.write("-10.000000000000000 1.000000000000000\n")
		new.write("0.000000000000000 " + str(change_factor) + "\n")
		new.write(str(source_sound_duration) + " " + str(change_factor)+ "\n")
		new.write(str(source_sound_duration) + " " + str(change_factor)+ "\n")

	if not extreme_precision:
		#use this parameters for good precision
		parameters =  "-t -Z  -S"+source_sound+" -Afft -M16384 -Np2 -M0.0238095000386238s -oversamp 8 -Whanning  -P0 -td_ampfac 1.20000004768372 -FCombineMul -shape 1 -D"+bpf_file +"  "+ target_sound
	else:
		#use this parameters for extreme precision
		parameters = "-t -Z  -S"+source_sound+" -Afft  -Np4 -M0.095238097012043s -oversamp 8 -Whanning  -P1 -td_thresh 2.90000009536743 -td_G 3.79999995231628 -td_band 0,4500.0078125 -td_nument 33.3333320617676 -td_minoff 0.02s  -td_mina 0 -td_minren 0.100000001490116 -td_evstre 1 -td_ampfac 1.20000004768372 -FCombineMul -shape 1 -D"+ bpf_file + " "+target_sound
	cmd 		= super_vp_path + " " + parameters
	args 		= shlex.split(cmd)
	p 			= subprocess.Popen(args)

	p.wait() #wait
	os.remove(bpf_file)


def stretch_sound_with_bpf_file(source_sound, target_sound, bpf_file, extreme_precision = True):
	"""
	stretch source sound following bpf file
	source_sound: audio source file
	target_sound: audio file to be created
	"""	
	from scikits.audiolab import wavread 
	x, fs, enc 		= wavread(str(source_sound))	

	if not extreme_precision:
		#use this parameters for good precision
		parameters =  "-t -Z  -S"+source_sound+" -Afft -M16384 -Np2 -M0.0238095000386238s -oversamp 8 -Whanning  -P0 -td_ampfac 1.20000004768372 -FCombineMul -shape 1 -D"+bpf_file +"  "+ target_sound
	else:
		#use this parameters for extreme precision
		parameters = "-t -Z  -S"+source_sound+" -Afft  -Np4 -M0.095238097012043s -oversamp 8 -Whanning  -P1 -td_thresh 2.90000009536743 -td_G 3.79999995231628 -td_band 0,4500.0078125 -td_nument 33.3333320617676 -td_minoff 0.02s  -td_mina 0 -td_minren 0.100000001490116 -td_evstre 1 -td_ampfac 1.20000004768372 -FCombineMul -shape 1 -D"+ bpf_file + " "+target_sound
	
	cmd 		= super_vp_path + " " + parameters
	args 		= shlex.split(cmd)
	p 			= subprocess.Popen(args)

	p.wait() #wait


## denoise sound file
def denoise_sound(source_sound, target_sound, gain_reduction =10, amp_threshold = -55, wnd_size = 8192 ):
	"""
	Denoise source_sound
	source_sound : audio source file
	target_sound : audio file to be created
	gain_reduction : gain reduction in dB to apply to the noise (must be positive)
	amp_threshold : amplitude to search for silent sounds
	wnd_size : when searching for silent parts in the sound, the size of moving window 
			for short sounds use small windows and viceversa
	"""
	import os
	from audio_analysis import find_silence, get_sound_duration
	#find silence
	silence_tags = find_silence(source_sound, threshold = amp_threshold, wnd_size=wnd_size)

	#get longer silence tag
	max_tag = 0
	for tag in silence_tags:
		if (tag[1]-tag[0]) > max_tag:
			max_tag = tag[1]-tag[0]
			beg = tag[0]
			end = tag[1] 

	#create noise key sdif
	noise_sdif = "aux_noise_sdif_13451.sdif"
	parameters = "-S"+source_sound+" -A -Np4 -M0.0500000007450581s -oversamp 8 -Whanning -avseg "+str(beg)+","+str(end)+" -OM2 "+noise_sdif

	cmd 		= super_vp_path + " " + parameters
	args 		= shlex.split(cmd)
	p 			= subprocess.Popen(args)
	p.wait() #wait

	sound_duration = str(get_sound_duration(source_sound))

	#denoise
	parameters =  "-t -A -Z -B0.000000 -E"+sound_duration+" -S"+source_sound+" -Np4 -M0.050000000745058s -oversamp 8 -Whanning -norm -Fsub "+noise_sdif+" -avgamma 1 -avbeta "+str(gain_reduction)+" -avsfac 0.000000000000000 "+target_sound
	cmd 		= super_vp_path + " " + parameters
	args 		= shlex.split(cmd)
	p 			= subprocess.Popen(args)

	p.wait() #wait

	#delete noise key
	os.remove(noise_sdif)

def transient_detection_analysis(audio_file, analysis = "", wait = True):
	if analysis== "":
		file_tag = os.path.basename(audio_file)
		file_tag = os.path.splitext(file_tag)[0]
		analysis = os.path.dirname(audio_file)+ "/" + file_tag + ".txt"

	parameters 	=  "-S"+audio_file+"  -A -Np4 -M0.0500000007450581s -oversamp 8 -Whanning -OT " + analysis
	cmd 		= super_vp_path + " " + parameters
	args 		= shlex.split(cmd)
	p 			= subprocess.Popen(args)	
	if wait:
		p.wait() #wait


	