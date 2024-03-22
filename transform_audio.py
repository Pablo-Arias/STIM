#-*- coding: utf-8 -*-
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ----------
# ---------- Made by pablo Arias @ircam on 02/2016
# ---------- Copyright (c) 2018 CREAM Lab // CNRS / IRCAM / Sorbonne Universite
# ----------
# ---------- execute different audio transformations
# ----------
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
from __future__ import absolute_import
from __future__ import print_function
from scipy.io.wavfile import read, write
import aifc
import glob
import numpy as np
from numpy import pi, polymul
import os
import soundfile
import sys
from conversions import lin2db, db2lin

def wav_to_mono(source, target):
	"""
	|Input:
	|	source : source audio file
	|	target : target audio file
	
	"""
	f = soundfile.SoundFile(source)

	if f.channels != 1:
		#To mono
		x, fs = soundfile.read(str(source))
		soundfile.write(target, x[:,0] , fs, f.subtype)


def mono_to_stereo(source, target):
	import numpy as np

	f = soundfile.SoundFile(source)
	if f.channels == 1:
		#To stereo
		x, fs = soundfile.read(str(source))
		soundfile.write(target, np.array([x, x]).transpose(), fs, f.subtype)

def format_to_format(source, target):
	"""
	source : source audio file
	target : target audio file
	"""
	x, fs = soundfile.read(str(source))		
	soundfile.write(target, x, fs)


def aif_to_wav(source, target):
	"""
	source : source audio file
	target : target audio file
	"""
	print('This functions will be deprecated, please use format_to_format instead')
	format_to_format(source, target)


def ogg_to_wav(source, target):
	"""
	source : source audio file
	target : target audio file
	"""
	print('This functions will be deprecated, please use format_to_format instead')
	format_to_format(source, target)


def wav_to_aif(source, target):
	"""
	source : fsource audio file
	target : starget audio file
	"""
	print('This functions will be deprecated, please use format_to_format instead')
	format_to_format(source, target)

def change_bit_encoding(source, target, new_pcm):
	"""
	source : source audio file
	target : target audio file
	new_pcm : 
	This function hasn't been tested
	"""
	
	x, fs = soundfile.read(str(source))		
	soundfile.write(target, x, fs, new_pcm)

def down_sample_audio_file(source, target, new_sf):
	from pydub import AudioSegment
	from conversions import get_file_without_path

	sound = AudioSegment.from_file(source)
	sound = sound.set_frame_rate(new_sf)
	sound = sound.set_sample_width(2)

	extension = get_file_without_path(source, with_extension=True).split(".")[1]
	sound.export(out_f=target, format=extension)


def combine_audio_files(audios, output):
	import subprocess
	import os
	
	command = "ffmpeg -i "+audios[0]+" -i "+audios[1]+" -filter_complex \"[0][1]amix=inputs=2[a]\" -map \"[a]\" "+output

	subprocess.call(command, shell=True)

	


##### ------------------------------------------------------	
##### ------------------------------------------------------
##### ------ Functions for preparing sounds for stimuli : 
##### ------ index_wav_file
##### ------ cut_silence_in_sound
##### ------------------------------------------------------
##### ------------------------------------------------------
def extract_sentences_tags(source, rms_threshold = -50, WndSize = 16384, overlap = 8192):
	"""
	This function extracts all the sentences inside an audiofile.
	
	Only works if sentences are at least 500 ms long, which can be tuned
	
	You can change the rms threshold to tune the algorithm

	input:
		source : fsource audio file
		rms_threshold : this is the threshold
		WndSize : window size to compue the RMS on
		overlap : nb of overlap samples

	returns:
		tags in pairs of [begining end]
	"""	
	x, fs = soundfile.read(str(source))

	index = 0
	NbofWrittendFiles = 1
	tags = []
	vid_lengths = []
	while index + WndSize < len(x):
		DataArray 	= x[index: index + WndSize]
		rms 		= np.sqrt(np.mean(np.absolute(DataArray)**2))
		rms 		= lin2db(rms)
		index 		= WndSize + index
		#index       += overlap
		if rms > rms_threshold:
			end 		= 0
			begining 	= index
			index 		= WndSize + index
			#index       += overlap
			while rms > rms_threshold:
				if index + WndSize < len(x):
					index 		= WndSize + index
					#index       += overlap
					DataArray 	= x[index: index + WndSize]
					rms 		= np.sqrt(np.mean(np.absolute(DataArray)**2))
					rms 		= lin2db(rms)
					end 		= index
				else:
					break

			
			#if file is over 500 ms long, write it
			if (end - begining) > (fs / 2) :
				begining = begining - WndSize
				if begining < 0: 
					begining = 0
				 
				end = end + WndSize
				if end > len(x): 
					end = len(x)

				#samples to seconds, minutes, hours 
				begining_s = begining/float(fs)
				end_s = end/float(fs)
				len_s = (end - begining) / float(fs)
				#print  "duree  :  "+ str(len_s)
				
				from datetime import timedelta, datetime
				begining_s = datetime(1,1,1) + timedelta(seconds=begining_s)
				end_s = datetime(1,1,1) + timedelta(seconds=end_s)
				len_s = datetime(1,1,1) + timedelta(seconds=len_s)
				begining_s = "%d:%d:%d.%3d" % (begining_s.hour, begining_s.minute, begining_s.second, begining_s.microsecond)
				end_s = "%d:%d:%d.%3d" % (end_s.hour, end_s.minute, end_s.second, end_s.microsecond)
				len_s = "%d:%d:%d.%3d" % (len_s.hour, len_s.minute,len_s.second , len_s.microsecond)

				#print "la longueur est"
				#print len_s

				tags.append([begining_s, end_s])
				vid_lengths.append(len_s)
				
				NbofWrittendFiles = NbofWrittendFiles + 1

	return tags, vid_lengths


def index_wav_file(source, rms_threshold = -50, WndSize = 16384, target_folder = "Indexed", add_time_tag=False):
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
	x, fs = soundfile.read(str(source))

	index = 0
	NbofWrittendFiles = 1
	while index + WndSize < len(x):
		DataArray 	= x[index: index + WndSize]
		rms 		= np.sqrt(np.mean(np.absolute(DataArray)**2))
		rms 		= lin2db(rms)
		index 		= WndSize + index
		if rms > rms_threshold:
			end 		= 0
			begining 	= index
			index 		= WndSize + index
			while rms > rms_threshold:
				if index + WndSize < len(x):
					index 		= WndSize + index
					DataArray 	= x[index: index + WndSize]
					rms 		= np.sqrt(np.mean(np.absolute(DataArray)**2))
					rms 		= lin2db(rms)
					end 		= index
				else:
					break

			
			#if file is over 500 ms long, write it
			if (end - begining) > (fs / 2) :
				duree = (end - begining) / float(fs)
				print("duree  :  "+ str(duree))
				
				begining = begining - WndSize
				if begining < 0: 
					begining = 0
				 
				end = end + WndSize
				if end > len(x): 
					end = len(x)

				sound_tag = os.path.basename(source)
				sound_tag = os.path.splitext(sound_tag)[0]
				try:
					os.mkdir(target_folder)
				except:
					pass

				#write(filename = target_folder+"/"+ sound_tag + "_" + str(NbofWrittendFiles)+".wav",rate = fs, data= x[begining:end])
				#wavwrite(x[begining:end], target_folder+"/"+ sound_tag + "_" + str(NbofWrittendFiles)+".wav", fs, enc='pcm24')
				if not add_time_tag:
					soundfile.write(target_folder+"/"+ sound_tag + "_" + str(NbofWrittendFiles)+".wav", x[begining:end] , fs)
				else:
					start_tag = np.round(begining/float(fs), 3)
					stop_tag  = np.round(end/float(fs), 3)
					name = target_folder+"/"+ sound_tag + "_" + str(NbofWrittendFiles)+ "_" + str(start_tag) + "_" + str(stop_tag)+".wav"
					soundfile.write(name, x[begining:end] , fs)

				NbofWrittendFiles = NbofWrittendFiles + 1

#High pass filter all the stimuli as there is too much noise in the bass frequency that is not letting me compute well the amplitude env.
def filter_file(s, src, target, type, freq, q =1):
	"""
	Filter type. : int, optional
		0 : lowpass (default)
		1 : highpass
		2 : bandpass
		3 : bandstop
		4 : allpass
	freq : float or PyoObject, optional, Cutoff or center frequency of the filter. Defaults to 1000.
	q : float or PyoObject, optional Q of the filter, defined (for bandpass filters) as freq/bandwidth. Should be between 1 and 500. Defaults to 1.
	
	Example: 
		from pyo import Server
		s  = Server(duplex=0, audio="offline")
		for file in glob.glob("*.wav"):
			filter_file(s, file, "filtered/"+file, type = 1, freq = 300, q = 1)
			s.shutdown()
	"""
	from pyo import sndinfo, SfPlayer, Biquad
	
	duration = sndinfo(src)[1]
	s.boot()
	s.recordOptions(dur = duration
	                , filename = target
	                , fileformat = 0
	                , sampletype = 0
	                )

	sf    = SfPlayer(src, speed = 1, loop = False)
	f = Biquad(sf, freq=freq, q=q, type=type).mix(2).out() # highpass
	s.start()
	s.stop()
	s.shutdown()

##
def cut_silence_in_sound(source, target, rmsTreshhold = -40, WndSize = 128):
	"""
	source : fsource audio file
	target : output sound
	This function cuts the silence at the begining and at the end of an audio file in order. 
	It's usefull for normalizing the length of the audio stimuli in an experiment.
	The default parameters were tested with notmal speech.
	"""	
	NbofWrittendFiles = 1
	x, fs = soundfile.read(str(source))
	index = int(0)

	
	#Remove the silence at the begining
	while index + WndSize < len(x):
		DataArray 	= x[index: index + WndSize]
		rms 		= np.sqrt(np.mean(np.absolute(DataArray)**2))
		rms 		= lin2db(rms)
		index 		= int(0.5*WndSize + index)

		if rms > rmsTreshhold:
			end 		= 0
			beginning 	= index
			break

	#Remove the silence at the end
	x, fs 		= soundfile.read(str(source))
	WndSize 		= 128
	index = 0
	x = list(reversed(x))

	while index + WndSize < len(x):
		DataArray 	= x[int(index): int(index + WndSize)]
		rms 		= np.sqrt(np.mean(np.absolute(DataArray)**2))
		rms 		= lin2db(rms)
		index 		= int(0.5*WndSize + index)

		if rms > rmsTreshhold:
			end 		= 0
			final 	= index
			break

	#write the sound source without silences

	x, fs 		     = soundfile.read(str(source))
	name_of_source   =  str(os.path.basename(source))
	name_of_source   = os.path.splitext(name_of_source)[0]
	path, sourcename = os.path.split(source)
	f = soundfile.SoundFile(str(source))
	soundfile.write(target, x[beginning:len(x)-final], fs, f.subtype)

def get_sound_without_silence(source, rmsTreshhold = -40, WndSize = 128):
	"""
	source : source audio file
	This function returns a begining and end time tags for the begining and the end of audio in a file
	"""	
	x, fs, enc 		= wavread(str(source))
	index = 0
	
	#Remove the silence at the begining
	while index + WndSize < len(x):
		DataArray 	= x[index: index + WndSize]
		rms 		= np.sqrt(np.mean(np.absolute(DataArray)**2))
		rms 		= lin2db(rms)
		index 		= 0.5*WndSize + index

		if rms > rmsTreshhold:
			end 		= 0
			beginning 	= index
			break

	#Remove the silence at the end
	x, fs, enc 		= wavread(str(source))
	WndSize 		= 128
	index = 0
	x = list(reversed(x))

	while index + WndSize < len(x):
		DataArray 	= x[int(index): int(index + WndSize)]
		rms 		= np.sqrt(np.mean(np.absolute(DataArray)**2))
		rms 		= lin2db(rms)
		index 		= 0.5*WndSize + index

		if rms > rmsTreshhold:
			end 		= 0
			final 	= index
			break

	#write the sound source without silences
	x, fs, enc 		= wavread(str(source))
	WndSize 		= 128
	rmsTreshhold 	= -70 
	index = 0

	end = len(x)-final
	return beginning/fs, end/fs


##### ------------------------------------------------------	
##### ------------------------------------------------------
##### ------ Normalize loudness
##### ------ This calls matlab backend to normalize
##### ------------------------------------------------------
##### ------------------------------------------------------
def normalize_folder_loudness(folder, target_folder, dbA=70):
	"""
	This functionc calls a matlab script for normalising the loudness of a folder
	folder : the folder where you have you files
	target_folder : the folder where you want your files to be saved
	dbA : normalisation threshold

	warning : only works with mono file and only reads wav files, for now
	This only works if you have matlab and matlab engine installed
	"""
	import sys
	import matlab.engine
	eng = matlab.engine.start_matlab()
	eng.addpath("/Users/arias/Documents/Developement/Matlab/",nargout=0)
	eng.normalize1(folder,target_folder, dbA , nargout=0)
	eng.quit()


def change_file_loudness(file, target_file, db_lufs=-15):
	"""
	|	This functionc calls a matlab script for normalising transforming a file to a target loudness
	|	folder : the folder where you have you files
	|	target_folder : the folder where you want your files to be saved
	|   db_lufs : target db_lufs for new file

	|	Warning : only works with mono file and only reads wav files, for now
	"""

	import soundfile as sf
	import pyloudnorm as pyln

	data, rate = sf.read(file) # load audio

	# measure the loudness first 
	meter    = pyln.Meter(rate) # create BS.1770 meter
	loudness = meter.integrated_loudness(data)

	# loudness normalize audio to -25 dB LUFS
	loudness_normalized_audio = pyln.normalize.loudness(data, loudness, db_lufs)

	sf.write(file = target_file, data=loudness_normalized_audio, samplerate=rate)


	#Old function using matlab engine
	#import sys
	#import matlab.engine
	#eng = matlab.engine.start_matlab()
	#eng.addpath("/Users/arias/Documents/Developement/Matlab/",nargout=0)
	#eng.file_to_loundness(file,target_file, dbA , nargout=0)
	#eng.quit()



def audio_peak_normalisation(source, target, dB=-1):
	"""
	|Input:
	|	source : input mono audio file
	|	target : target file
	|
	|	This is a quick and simple peak normalsiation made by Pablo Arias
	| 	Peak normalize audio to X dB

	"""
	import pyloudnorm as pyln

	x, fs = soundfile.read(str(source))
	f = soundfile.SoundFile(str(source))
	
	peak_normalized_audio = pyln.normalize.peak(x, dB)
	
	soundfile.write(target, peak_normalized_audio, fs, f.subtype)


def apply_fade(source, target, duration=0.05, fade_type = "out"):
	"""
	|This functions create either fades in or outs in mono and stereo sounds.
	|Inputs:
	|	source : audio file to transform 
	|	target : target audio file to create
	|	duration : duration of the fade in seconds 
	|	fade_type : either "in" or "out"
	|
	|Output:
	|	No output. Create a target audio file.
	"""
	# convert to audio indices (samples)
	from audio_analysis import get_nb_channels

	x, fs = soundfile.read(str(source))
	f = soundfile.SoundFile(source)

	length = int(round(duration*fs))

	# compute fade curve
	if fade_type == "in":
		#Define fade in interval
		end   = length
		start = 0

		# linear fade
		fade_curve = np.linspace(0, 1.0, length)
	
	elif fade_type == "out":
		#Define fade out interval
		end = x.shape[0]
		start = end - length
		fade_curve = np.linspace(1.0, 0.0, length)
	else:
		exit("fade_type must be either 'in' or 'out'")

	# apply the curve
	nb_channels = get_nb_channels(source)

	if nb_channels == 2:
		#Separate channels
		ch_1 = x[:,0]
		ch_2 = x[:,1]

		#Apply curve
		ch_1[start:end] = ch_1[start:end] * fade_curve
		ch_2[start:end] = ch_2[start:end] * fade_curve

		#prepare output
		out = np.array([ch_1, ch_2])
		out = np.transpose(out)
	
	elif nb_channels == 1:
		x[start:end] = x[start:end] * fade_curve
		out = x
	
	else:
		exit("This function only supports stereo and mono files")


	soundfile.write(target, out, fs, f.subtype)


