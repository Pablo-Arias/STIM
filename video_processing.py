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

def get_movie_duration(video):
	"""
	Get the duration of video

	parameters:
		video : videon input
	"""	
	import subprocess
	import os
	import shlex, subprocess

	command = "ffmpeg -i "+video+" 2>&1 | grep \"Duration\""
	output = subprocess.check_output(command, shell=True)
	return output
	
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
	"""

	import subprocess
	import os

	#command = "ffmpeg -i "+source_name+" -ss "+start+" -t "+end+" -async 1 "+target_name
	command = "ffmpeg -i "+source_name+" -ss "+start+" -t "+length+" -async 1 -strict -2 "+target_name

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
	command = "ffmpeg -i "+source+" -vcodec mpeg4 -vtag XVID -b 990k -bf 2 -g 300 -s 720x576 -acodec libmp3lame  -pass 1 -ab 256 -threads 0 -f avi "+target
	subprocess.call(command, shell=True)


def extract_frames_video(source, folder, tag=""):
	"""
	Extract the frames of a video
	"""
	import subprocess
	import os

	os.mkdir(folder)

	command = "ffmpeg -i "+source+" -r 4 "+folder+tag+"$filename%01d.bmp"
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
	print("python3")
	import subprocess
	import os
	import shlex
	command = "ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate " + source
	#output  = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).communicate()[0]

	#subprocess.call(command, shell=True)

	#p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	#out, err =  p.communicate()
	#return out

	#p = subprocess.Popen(shlex.split(command), bufsize=1, universal_newlines=True)
	#return p.communicate()
	from subprocess import check_output

	x = check_output(["ffprobe", "-v", "error", "-select_streams", "v", "-of", "default=noprint_wrappers=1:nokey=1", "-show_entries stream=r_frame_rate", source],shell=True,stderr=subprocess.STDOUT)

	return x


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
	video_codec : specifiy the video copy flag to pass to ffmpef. 
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


