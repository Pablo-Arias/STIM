# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ---------- Made by pablo Arias @ircam on 02/2016
# ---------- execute different audio transformations
# ---------- to use this don't forget to include these lines before your script:
# ----------
# ---------- import sys
# ---------- sys.path.append('/Users/arias/Documents/Developement/Python/')
# ---------- from transform_audio import 'the functions you need'
# ----------
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#


from scipy.io.wavfile import read, write
from scikits.audiolab import aiffwrite, aiffread, wavwrite, wavread, Sndfile, oggread  # scipy.io.wavfile can't read 24-bit WAV
from scikits.audiolab import aiffread, aiffwrite # scipy.io.wavfile can't read 24-bit WAV
import aifc
from pyo import *
import glob
from scipy.io.wavfile import read, write

import aifc
import numpy as np
from numpy import pi, polymul
import os

import sys
sys.path.append('/Users/arias/Documents/Developement/Python/')
from conversions import lin2db, db2lin

def wav_to_mono(source, target):
	"""
	source : source audio file
	target : target audio file
	"""
	f = Sndfile(source, 'r')
	if f.channels != 1:
		#To mono
		x, fs, enc = wavread(source)
		f.channels
		wavwrite(x[:,0], target, fs, enc='pcm24')


def mono_to_stereo(source, target):
	import numpy as np
	f = Sndfile(source, 'r')
	if f.channels == 1:
		
		#To mono
		x, fs, enc = wavread(source)
		print "here"
		print type(x)
		print type(np.array([x, x]))
		wavwrite(np.array([x, x]).transpose(), target, fs, enc='pcm24')



def aif_to_wav(source, target):
	"""
	source : source audio file
	target : target audio file
	"""
	try: 
		x, fs, enc 		= aiffread(str(source))
		WavFileName = target
		wavwrite(x, WavFileName, fs, enc='pcm24')
	except:
		print "File is not aif"
		pass

def ogg_to_wav(source, target):
	"""
	source : source audio file
	target : target audio file
	"""
	x, fs, enc = oggread(source)
	WavFileName = target
	wavwrite(x, WavFileName, fs, enc='pcm24')

def wav_to_aif(source, target):
	"""
	source : fsource audio file
	target : starget audio file
	"""
	x, fs, enc 		= wavread(str(file))
	AifFileName = target
	wavwrite(x, AifFileName, fs, enc='pcm24')



##### ------------------------------------------------------	
##### ------------------------------------------------------
##### ------ Functions for preparing sounds for stimuli : 
##### ------ index_wav_file
##### ------ cut_silence_in_sound
##### ------------------------------------------------------
##### ------------------------------------------------------
def extract_sentences_tags(source, rms_threshold = -50, WndSize = 16384, overlap = 8192):
	"""
	This function separates all the sentences inside an audiofile.
	It takes each sentence and put it into one audio file inside target_folder with the name target_nb
	The default parameters were tested with notmal speech.
	Only works if file is at least 500 ms long, which can be tuned
	You can change the rms threshold to tune the algorithm

	input:
		source : fsource audio file
		rms_threshold : this is the threshold
		WndSize : window size to compue the RMS on
		overlap : nb of overlap samples

	returns:
		tags in pairs of [begining end]
	"""	
	try:
		x, fs, enc 		= aiffread(str(source))
	except:
		x, fs, enc 		= wavread(str(source))

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
 				print  "duree  :  "+ str(len_s)
 				
 				from datetime import timedelta, datetime
 				begining_s = datetime(1,1,1) + timedelta(seconds=begining_s)
 				end_s = datetime(1,1,1) + timedelta(seconds=end_s)
 				len_s = datetime(1,1,1) + timedelta(seconds=len_s)
 				begining_s = "%d:%d:%d.%3d" % (begining_s.hour, begining_s.minute, begining_s.second, begining_s.microsecond)
 				end_s = "%d:%d:%d.%3d" % (end_s.hour, end_s.minute, end_s.second, end_s.microsecond)
 				len_s = "%d:%d:%d.%3d" % (len_s.hour, len_s.minute,len_s.second , len_s.microsecond)

 				print "la longueur est"
 				print len_s

 				tags.append([begining_s, end_s])
 				vid_lengths.append(len_s)
 				
 				NbofWrittendFiles = NbofWrittendFiles + 1

 	return tags, vid_lengths


def index_wav_file(source, rms_threshold = -50, WndSize = 16384, target_folder = "Indexed"):
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
	try:
		x, fs, enc 		= aiffread(str(source))
	except:
		x, fs, enc 		= wavread(str(source))

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
 				print  "duree  :  "+ str(duree)
 				
 				begining = begining - WndSize
 				if begining < 0: 
 					begining = 0
 				 
 				end = end + WndSize
 				if end > len(x): 
 					end = len(x)
 				print file
				sound_tag = os.path.basename(source)
				sound_tag = os.path.splitext(sound_tag)[0]
				try:
					os.mkdir(target_folder)
				except:
					pass

 				#write(filename = target_folder+"/"+ sound_tag + "_" + str(NbofWrittendFiles)+".wav",rate = fs, data= x[begining:end])
 				wavwrite(x[begining:end], target_folder+"/"+ sound_tag + "_" + str(NbofWrittendFiles)+".wav", fs, enc='pcm24')
 				NbofWrittendFiles = NbofWrittendFiles + 1

#High pass filter all the stimuli as there is too much noise in the bass frequency that is not letting me compute well the amplitude env.
def filter_file(s, src, target, type, freq, q =1):
	"""
	freq : float or PyoObject, optional
	Cutoff or center frequency of the filter. Defaults to 1000.
	q : float or PyoObject, optional
	Q of the filter, defined (for bandpass filters) as freq/bandwidth. Should be between 1 and 500. Defaults to 1.
	type : int, optional
	Filter type. Five possible values :
		0 : lowpass (default)
		1 : highpass
		2 : bandpass
		3 : bandstop
		4 : allpass

	example of use: 
		s  = Server(duplex=0, audio="offline")
		for file in glob.glob("*.wav"):
			filter_file(s, file, "filtered/"+file, type = 1, freq = 300, q = 1)
			s.shutdown()
	"""
	



	from pyo import sndinfo
	
	
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
			print beginning/44100
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
			print (len(x)-final)/44100
			break

	#write the sound source without silences
	x, fs, enc 		= wavread(str(source))
	WndSize 		= 128
	rmsTreshhold 	= -70 
	index = 0

 	name_of_source =  str(os.path.basename(source))
 	name_of_source = os.path.splitext(name_of_source)[0]
 	path, sourcename = os.path.split(source)
 	wavwrite(x[beginning:len(x)-final], target, fs, enc='pcm24')

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


def change_file_loudness(file, target_file, dbA=70):
	"""
	This functionc calls a matlab script for normalising transforming a file to a target loudness
	folder : the folder where you have you files
	target_folder : the folder where you want your files to be saved
	dbA : target dbA for new file

	warning : only works with mono file and only reads wav files, for now
	"""
	import sys
	import matlab.engine
	eng = matlab.engine.start_matlab()
	eng.addpath("/Users/arias/Documents/Developement/Matlab/",nargout=0)
	eng.file_to_loundness(file,target_file, dbA , nargout=0)
	eng.quit()
	


