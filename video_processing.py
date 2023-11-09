#-*- coding: utf-8 -*-
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ---------- Made by pablo Arias @ircam on 11/2015
# ---------- Copyright (c) 2018 CREAM Lab // CNRS / IRCAM / Sorbonne Université
# ----------
# ---------- process video in diferent ways
# ---------- to use this don't forget to include these lines before your script:
# ----------
# ---------- 
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#

from __future__ import absolute_import
from __future__ import print_function
import sys
from transform_audio import extract_sentences_tags
import cv2
from datetime import datetime
import os
import glob
from conversions import get_file_without_path
from transform_audio import index_wav_file

def extract_audio(video_file, target_name):
	"""
	parameters:
		video_file : file to extract wav from

	extract audio to wav and return the name of the extracted audio
	"""
	import subprocess
	import os
	
	try:
		os.remove(target_name)
	except:
		pass

	base = os.path.basename(video_file)
	file_name = os.path.splitext(base)[0]
	path = os.path.dirname(os.path.realpath(video_file))


	command = "ffmpeg -i "+video_file+" -ab 160k -ac 2 -ar 44100 -vn "+ target_name
	subprocess.call(command, shell=True)

	return target_name


def replace_audio(video_file, new_audio, target_video):
	"""
	Replace the audio of video_file with new_audio

	parameters:
		video_file : video file to change the audio
		new_audio : new audio to use
		target_video : video to be created
	"""
	import subprocess
	import os

	#remove target video if it exists
	try:
		os.remove(target_video)
	except:
		pass

	base = os.path.basename(video_file)
	file_name = os.path.splitext(base)[0]
	path = os.path.dirname(os.path.realpath(video_file))

	
	#Old command to process /mov, re-encoding audio
	#command = "ffmpeg -i "+video_file+" -i "+new_audio+" -map 0:v -map 1:a -c copy -shortest " + target_video
	
	#New command to process mp4, re-encoding audio
	command = "ffmpeg -i "+video_file+" -i "+new_audio+" -c:v copy -map 0:v:0 -map 1:a:0 "+target_video

	subprocess.call(command, shell=True)


def erase_audio_from_video(input_video, output_video):
	"""
	Erase the audio from a video

	parameters:
		input_video : source video
		output_video : video to be created without the audio
		
	"""	
	import subprocess
	import os

	command = "ffmpeg -i "+input_video+" -vcodec copy -an "+output_video
	subprocess.call(command, shell=True)
	


def get_movie_stream(video):
	"""
	Get movie stream

	parameters:
		video : video from which stream will be extracted		
	"""		
	import subprocess
	import os
	import shlex, subprocess

	command = "ffmpeg -i "+video+" 2>&1 | grep \"Stream\""
	output = subprocess.check_output(command, shell=True)
	return output



def get_movie_duration(video, in_seconds=True):
	"""
	Get the duration of video

	parameters:
		video : videon input
	"""	
	import subprocess
	import os
	import shlex, subprocess

	command = "ffmpeg -i "+video+" 2>&1 | grep \"Duration\""
	result = subprocess.check_output(command, shell=True)

	output      = str(result)
	f_dur       = output.split(" ")[3][:-1]

	if f_dur == "N/A":
		result = 0
		
	elif in_seconds:	
		f_dur       = datetime.strptime(f_dur, "%H:%M:%S.%f")
		a_timedelta = f_dur - datetime(1900, 1, 1)
		result      = a_timedelta.total_seconds()
	
	return result

	
def extract_sentences_in_video(source_name, target_folder, rms_threshold = -50, WndSize = 16384, overlap = 8192):
	"""
	In a video with many sentences and blanks between the sentences, create separate video files with one sentence per file.
	This is very usefull for indexing databases.
	This only works if the video has sound.
	The algorithm looks at the RMS of the audio and extracts the sentences.

	parameters:
		source_name : videon input
		target_folder : folder in which the videos will be cretaed
		rms_threshold : rms_threshold specifying where to consider there is a slience
		WndSize :
		overlap :
	"""	
	import os
	
	audio_name = "auxiliary_audio_file_14357XXX.wav"
	try:
		os.remove(audio_name)
	except:
		pass

	file_name = os.path.splitext(source_name)[0]
	file_name = os.path.basename(file_name)

	extract_audio(source_name, audio_name)
	tags, lengths = extract_sentences_tags(audio_name, rms_threshold = rms_threshold, WndSize = WndSize, overlap = overlap)

	
	cpt=1
	for tag in tags:
		extract_sub_video_sentences( source_name = source_name
										, target_name = target_folder +file_name+"_"+ str(cpt)+ ".mp4"
										, start = tag[0]
										, length = lengths[0]
									)
		cpt+=1

	os.remove(audio_name)


def extract_sub_video_sentences(source_name, target_name, start, length):
	"""
	Takes a sub video in source_name begining at start and ending at end
	The start time should be in this format: 00:24:00
	Length in seconds.

	To format start you may wqnt to use:
	> import datetime
	> str(datetime.timedelta(seconds=666))
	> '0:11:06'
	"""

	import subprocess
	import os

	#command = "ffmpeg -i "+source_name+" -ss "+start+" -t "+end+" -async 1 "+target_name
	command = "ffmpeg -i "+source_name+" -ss "+str(start)+" -t "+str(length)+" -async 1 -q:v 1 -strict -2 "+target_name

	subprocess.call(command, shell=True)

def video_scaling(source_name, target_name, resolution):
	"""
	changes the scale of the video
	source name : video to be transformed 
	target name : transformed video
	resolution : target resolution, tuple, pair of values
	"""
	import subprocess
	import os
	try:
		os.remove(target_name)
	except:
		pass	

	resolution = [str(i) for i in resolution]
	command = "ffmpeg -i "+source_name+" -vf scale="+resolution[0]+":"+resolution[1]+":force_original_aspect_ratio=decrease -strict -2 "+target_name+" -hide_banner"
	subprocess.call(command, shell=True)


def four_3_to_16_9(source_name, target_name):
	"""
	changes the scale of the video
	source name : video to be transformed 
	target name : transformed video
	resolution : target resolution, tuple, pair of values
	"""
	import subprocess
	import os
	try:
		os.remove(target_name)
	except:
		pass	

	#command = "ffmpeg -i "+source_name+" -vf \"scale=640x480,setsar=1,pad=854:480:107:0\" "+target_name
	command = "ffmpeg -i "+source_name+" -vf scale=1080x1920,setdar=16:9 "+target_name
	#command = "ffmpeg -i "+source_name+" -vf scale=1080x1920,setdar=16:9 "+target_name
	subprocess.call(command, shell=True)

def four_3_to_16_9_v2(source_name, target_name):
	import subprocess
	import os
	try:
		os.remove(target_name)
	except:
		pass	

	#command = "ffmpeg -i "+source_name+" -vf \"scale=640x480,setsar=1,pad=854:480:107:0\" "+target_name
	command = "ffmpeg -i "+source_name+" -q:v 10 -g 1 "+target_name
	subprocess.call(command, shell=True)

def cut_silence_in_video(source, target, rmsTreshhold = -40, WndSize = 128):
	"""
	cut the silence at the begining and at the end of the video
	source name : video to be transformed 
	target name : transformed video
	rmsTreshhold : threshold to use for silence
	WndSize : window size to search for silence at the begining and at the end
	"""	
	import os

	try:
		os.remove(target)
	except:
		pass

	#extract audio
	audio_aux = "aux_audio_15452.wav"
	extract_audio(source, audio_aux)


	#Extract_time tags for begining and end
	from .transform_audio import get_sound_without_silence
	begining_s, end_s = get_sound_without_silence(audio_aux, rmsTreshhold = rmsTreshhold, WndSize = WndSize)
	len_s = end_s - begining_s

	#adapt the time to the good format
	from datetime import timedelta, datetime
	begining_s = datetime(1,1,1) + timedelta(seconds=begining_s)
	end_s = datetime(1,1,1) + timedelta(seconds=end_s)
	len_s = datetime(1,1,1) + timedelta(seconds=len_s)
	begining_s = "%d:%d:%d.%3d" % (begining_s.hour, begining_s.minute, begining_s.second, begining_s.microsecond)
	end_s = "%d:%d:%d.%3d" % (end_s.hour, end_s.minute, end_s.second, end_s.microsecond)
	len_s = "%d:%d:%d.%3d" % (len_s.hour, len_s.minute,len_s.second , len_s.microsecond)
	
	print(begining_s)
	print(end_s)
	print(len_s)

	#extract sub_video
	extract_sub_video_sentences(source, target, begining_s, len_s)

	#remove aux audio
	os.remove(audio_aux)

def denoise_audio_in_video(source, target, gain_reduction =10, amp_threshold = -55, wnd_size = 8192  ):
	"""
	denois the sound from a  video
	source : video to be transformed 
	target : transformed video
	amp_threshold : threshold to use for silence
	gain_reduction : gain to reduce the noise
	wnd_size : moving window size
	"""
	from super_vp_commands import denoise_sound

	import os
	try:
		os.remove(target)
	except:
		pass

	#extract audio
	audio_aux = "aux_audio_1.wav"
	extract_audio(source, audio_aux)

	#denoise sound
	audio_aux2 = "aux_audio_2.wav"
	denoise_sound(audio_aux, audio_aux2
					, gain_reduction = gain_reduction
					, amp_threshold = amp_threshold
					, wnd_size = wnd_size 
				)

	#replace sound
	replace_audio(
		  video_file   = source
		, new_audio    = audio_aux2
		, target_video =  target
		)

	#remove sounds
	os.remove(audio_aux)
	os.remove(audio_aux2)

def change_video_format(source, target):
	"""
	Change the video extension of video to the one of file
	"""
	import subprocess
	import os
	try:
		os.remove(target)
	except:
		pass

	#command = "ffmpeg -i "+source+" -vcodec copy -acodec copy "+target
	command = "ffmpeg -i "+source+" -ar 22050 -b 3298k "+target
	subprocess.call(command, shell=True)



def convert_to_avi(source, target):
	import subprocess
	import os
	try:
		os.remove(target)
	except:
		pass
	
	#command = "ffmpeg -i "+source+" -vcodec mpeg4 -vtag XVID -b 990k -bf 2 -g 300 -s 640x360 -pass 1 -an -threads 0 -f rawvideo -y /dev/null"
	#subprocess.call(command, shell=True)


	#command = "ffmpeg -i "+source+" -vcodec mpeg4 -vtag XVID -b 990k -bf 2 -g 300 -s 720x576 -acodec libmp3lame -ab 256k -ar 48000 -ac 2 -pass 2 -threads 0 -f avi "+target
	command = "ffmpeg -i "+source+" -vcodec mpeg4  -af aresample=async=1000 -vtag XVID -b 990k -bf 2 -g 300 -s 720x576 -acodec libmp3lame  -pass 1 -ab 256 -threads 0 -f avi "+target
	subprocess.call(command, shell=True)


def change_frame_rate(source, target_fps, output):
	import subprocess
	import os
	command = "ffmpeg -i "+source+" -avoid_negative_ts make_zero -af apad -q:v 1 -af aresample=async=1000 -filter:v fps="+str(target_fps) +" " +output
	subprocess.call(command, shell=True)


def crop_video(source_video, target_video, x=0, y=0,out_w=0 , out_h=0 ):
	import subprocess
	import os
	#Something like this might work: 
	command = "ffmpeg -i "+source_video+" -strict -2 -filter:v crop="+str(out_w)+":"+str(out_h)+":"+str(x)+":"+str(y)+" "+target_video
	subprocess.call(command, shell=True)

def get_fps(source):
	"""
		Get fps of video file
	"""
	import subprocess
	import os
	import shlex
	command = "ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate " + source
	output  = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).communicate()[0]

	return output

	#subprocess.call(command, shell=True)

	#p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	#out, err =  p.communicate()
	#return out

	#p = subprocess.Popen(shlex.split(command), bufsize=1, universal_newlines=True)
	#return p.communicate()
	#from subprocess import check_output

	#x = check_output(["ffprobe", "-v", "error", "-select_streams", "v", "-of", "default=noprint_wrappers=1:nokey=1", "-show_entries stream=r_frame_rate", source],shell=True,stderr=subprocess.STDOUT)

	#return x


def color_correction(source_video, target_video, gamma=1.5, saturation =1.3):
	"""
	Gamma increases ...
	Saturation increases colors
	"""
	import subprocess
	import os
	
	command = "ffmpeg -i " + source_video + " -vf eq=gamma="+str(1.5)+":saturation="+str(saturation)+" -c:a copy "+ target_video
	subprocess.call(command, shell=True)

def create_movie_from_frames(frame_name_tag, fps, img_extension , target_video, video_codec="copy", preset="ultrafast", loseless=0):
	"""
	Create a movie with a series of frames
	frame_name_tag : if your frames are named frame_001.png, frame_name_tag="frame_"
	target_video : the video that will be created
	video_codec : specifiy the video copy flag to pass to ffmpeg. 

	Possiblities:
		"copy" : will copy frames, videos will be very big, but generation will bef ast
		"libx265" : generation will be slow but videos will be small
		"preset" : The default is medium. The preset determines compression efficiency and therefore affects encoding speed. 
				   options: superfast, veryfast, faster, fast, medium, slow, slower, veryslow, and placebo. 
				   Use the slowest preset you have patience for. Ignore placebo as it provides insignificant returns for a significant increase in encoding time.
		"loseless" : 1 : losseless encoding; 0 : loss encoding

 	"""
	import subprocess
	import os
	
	#command = "ffmpeg -framerate "+str(fps)+" -i "+frame_name_tag+"%d"+img_extension+" -vcodec "+video_codec+" -acodec copy -preset ultrafast "+target_video
	command = "ffmpeg -framerate "+str(fps)+" -i "+frame_name_tag+"%d"+img_extension+" -vcodec "+video_codec+" -acodec copy -preset "+preset+" -x265-params lossless="+str(loseless)+" "+target_video
	subprocess.call(command, shell=True)	


def compress_to_h265(video_in, video_out):
	
	import subprocess
	import os

	command = "ffmpeg -i "+video_in+" -vcodec libx265 -crf 28 "+video_out
	subprocess.call(command, shell=True)	
	

def sharpen_video(source_video, target_video):
	"""
	Create a movie with a series of frames
	frame_name_tag : if your frames are named frame_001.png, frame_name_tag="frame_"
	target_video : the video that will be created
	"""
	import subprocess
	import os
	
	command = "ffmpeg -i "+source_video+" -vf unsharp "+target_video
	subprocess.call(command, shell=True)	

def combine_2_videos(left, right, output, combine_audio_flag=True):
	"""
		Create a movie with 2 movies

	"""
	import subprocess

	if combine_audio_flag:
		import uuid 
		audio_1 = str(uuid.uuid1()) +"____.wav"
		audio_2 = str(uuid.uuid1()) +"____.wav"
		master_audio = str(uuid.uuid1()) +"____.wav"
		extract_audio(left , audio_1)
		extract_audio(right, audio_2)
		combine_audio([audio_1, audio_2], master_audio)


	#command = "ffmpeg -i "+left+" -i "+right+" -filter_complex \"[0][1]scale2ref=w=oh*mdar:h=ih[left][right];[left][right]hstack\" "+output

	#command = "ffmpeg -i "+left+" -i "+ right+" -filter_complex \"[0]scale=175:100:force_original_aspect_ratio=decrease,pad=175:100:-1:-1:color=gray,setsar=1[left];[1]scale=175:100:force_original_aspect_ratio=decrease,pad=175:100:-1:-1:color=gray,setsar=1[right];[left][right]hstack\" "+ output
	if combine_audio_flag:
		command = "ffmpeg -i "+left+" -i "+right+" -i "+master_audio+" -filter_complex \"[0][1]scale2ref=w=oh*mdar:h=ih[left][right];[left][right]hstack\" -map 2:a -c:a copy "+ output

	else:
		command = "ffmpeg -i "+left+" -i "+right+" -filter_complex \"[0][1]scale2ref=w=oh*mdar:h=ih[left][right];[left][right]hstack\" "+ output

	subprocess.call(command, shell=True)

	if combine_audio_flag:
		os.remove(master_audio)
		os.remove(audio_1)
		os.remove(audio_2)

def combine_videos( tl, tr, bl, br,output, audios = [] ):
	"""
		Create a movie with 4 movies!
		tl : top left; tr : top right, bl : bottom left; br : bottom_right

	"""
	import subprocess
	import os
	
	if audios != []:
		master_audio = "master_audio_aux_file.wav"
		combine_audio(audios, master_audio)
		#command = "ffmpeg -i "+tl+" -i "+tr+" -i "+bl+" -i "+bl+" -i "+ master_audio +" -filter_complex \"[0:v][1:v]hstack[t];[2:v][3:v]hstack[b];[t][b]vstack[v]\" -map \"[v]\" -c:a copy -shortest "+ output

		command = "ffmpeg -i "+tl+" -i "+tr+" -i "+bl+" -i "+br+" -i "+master_audio+" -filter_complex \"[0:v][1:v]hstack[t];[2:v][3:v]hstack[b];[t][b]vstack[v]\" -map \"[v]\" -map 4:a -c:a aac -shortest "+output

	else :
		command = "ffmpeg -i "+tl+" -i "+tr+" -i "+bl+" -i "+br+" -ac 2 -filter_complex \"[0:v][1:v]hstack[t];[2:v][3:v]hstack[b];[t][b]vstack[v]\" -map \"[v]\" -c:a aac -shortest "+ output

	subprocess.call(command, shell=True)

	#delete master audio
	if audios != []:
		os.remove(master_audio)


def combine_audio(files, target_audio, pre_normalisation=True):
    """
        input : file names in an array (you can use videos!!)
        Combine audio_files into one
    """
    import soundfile
    from transform_audio import wav_to_mono
    import os
    import numpy as np
    import uuid

    #Extract audio from video and convert to mono
    audio_files = []
    for file in files:
        #extract audio
		
        audio = str(uuid.uuid4()) + ".wav"
        audio_files.append(audio)
        print(audio)

        extract_audio(file, audio)
        import time
        time.sleep(2)

        #To mono
        wav_to_mono(audio, audio)
    
    #read audios
    raw_audios = []
    for file in audio_files:
        #read audio
        x, fs = soundfile.read(file)
        
        #normalize loudness, if needed
        if pre_normalisation:
            x = x / np.max(x)
            
        raw_audios.append(x)

    #Pad difference
    lengths = [len(i) for i in raw_audios]
    
    #Find longer file
    max_value = max(lengths)
    max_index = lengths.index(max_value)
    
    #pad audio
    paded_audio = []
    for raw_audio in raw_audios:
        diff = abs(len(raw_audio) - max_value)
        pad = [0.0 for i in range(diff)]
        pad = np.asarray(pad)
        paded_audio.append(np.concatenate([raw_audio, pad]))
    
    paded_audio = np.sum(paded_audio, axis=0)
    
    #normalize
    paded_audio = paded_audio/ np.max(paded_audio)
    
    #Export audio
    soundfile.write(target_audio, paded_audio , fs)
    
    #delete files
    for file in audio_files:
        os.remove(file)

def extract_frames_video(source, folder, tag="", fps=25, extension=".bmp", quality=2):
	"""
	Extract the frames of a video
	"""
	import subprocess
	import os

	os.mkdir(folder)

	command = "ffmpeg -i "+source+" -r "+str(fps)+" "+folder+tag+"$filename%01d"+extension +" -q:v "+str(quality) +" -qmin "+ str(1)
	subprocess.call(command, shell=True)


def open_cap(file)       : return cv2.VideoCapture(file)



def extract_frame(video, target_file, frame_nb):     
	"""
		Extract frame from a video.
		To extract several frames at once see extract_frames
	"""
	import matplotlib.pyplot as plt

	cap   = open_cap(video)
	cap.set(1,frame_nb)
	ret, frame = cap.read()
	frames.append(frame)
    
	#write frame
	ax = plt.subplot()
	plt.axis('on')
	ax = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
	ax.axes.get_xaxis().set_visible(False)
	ax.axes.get_yaxis().set_visible(False)

	#save fig
	plt.savefig(target_file)



def extract_frames(video, target_folder, frame_nbs = []):
	"""
	| Description: 
	| 	Extract frames specified in the frame_nbs array, from a video into a target folder.
		
	"""
	import matplotlib.pyplot as plt	

	cap   = open_cap(video)
	
	for frame_nb in frame_nbs:
		cap.set(1,frame_nb)
		ret, frame = cap.read()
	    
		#write frame
		plt.axis('on')
		ax = plt.subplot()
		ax = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		ax.axes.get_xaxis().set_visible(False)
		ax.axes.get_yaxis().set_visible(False)

		#save fig
		plt.savefig(target_folder + str(frame_nb) + ".jpg")



## Index interactions
def index_video_recording(source, rms_threshold = -50, WndSize = 16384, target_folder = "Indexed/"):
    target_file = target_folder + get_file_without_path(source) + ".wav"
    indexed_folder = target_folder + get_file_without_path(source) + "/"
    extract_audio(source, target_file)
    index_wav_file(target_file, rms_threshold = rms_threshold, WndSize = WndSize, target_folder = indexed_folder)


def index_video_recordings_parallel(sources, rms_threshold = -50, WndSize = 16384, indexed_path="extracted_audio/"):
    try:
        os.mkdir(indexed_path)
    except:
        pass    
    
    import multiprocessing
    from itertools import repeat
    pool_obj = multiprocessing.Pool()
    sources     = glob.glob(sources)

    pool_obj.starmap(index_video_recording, zip(sources
                                            , repeat(rms_threshold)
                                            , repeat(WndSize)
                                            , repeat(indexed_path)
                                            ))
    



