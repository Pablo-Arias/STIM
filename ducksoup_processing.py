#-*- coding: utf-8 -*-
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ---------- Made by pablo Arias @ircam on 11/2021
# ---------- Copyright (c) 2021 LUCS // CNRS / IRCAM / Sorbonne Université
# ----------
# ----------
# ---------- 
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#

from __future__ import absolute_import
from __future__ import print_function
import sys
from transform_audio import extract_sentences_tags
import glob
from conversions import get_file_without_path
from video_processing import replace_audio

def pair_videos(source_folder, target_folder, video_extension, audio_extension, pairing_style = {"video": "fx", "audio" : "in"}):
	"""
	pairing_style allow you to pair either neutral and manipulated audio with either neutral and manipulated video.
	For example, to paur manipulated video with manipulated audio use pairing_style = {"video": "fx", "audio" : "in"})

	Usage example : pair_videos(source_folder = "ducksoup/mirror/", target_folder = "ducksoup/paired/", video_extension = ".mts", audio_extension = ".ogg")

	"""
	for file in glob.glob(source_folder+"*"+pairing_style["video"]+video_extension):
		file_tag = get_file_without_path(file)
		date, hour, _, part_id, _, other_id, _, _, input_type, manip = file_tag.split("-")

		#corresponding audio
		corresp_audio = glob.glob(source_folder + "/*"+part_id+"*"+other_id+ "*"+pairing_style["audio"]+audio_extension)
		if len(corresp_audio) != 1:
			print("Check corresponding audio file for video  : "+ file)
			print("Not processing that video")
			continue
		corresp_audio = corresp_audio[0]
		replace_audio(file, corresp_audio, target_folder + get_file_without_path(file, with_extension=True))


		
