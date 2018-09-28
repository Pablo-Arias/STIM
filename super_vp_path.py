#-*- coding: utf-8 -*-
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ---------- Made by pablo Arias @ircam on 11/2015
# ---------- Copyright (c) 2018 CREAM Lab // CNRS / IRCAM / Sorbonne Universite
# ----------
# ---------- Analyse audio and return soudn features
# ---------- to us this don't forget to include these lines before your script:
# ----------
# ---------- import sys
# ---------- sys.path.append('/Users/arias/Documents/Developement/Python/')
# ---------- from parse_sdif import 'the functions you need'
# ----------
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
def get_super_vp_path():
	import glob, os
	AudioSculpt_version = glob.glob("/Applications/AudioSculpt*/AudioSculpt*.app/Contents/MacOS/supervp")[0]
	AudioSculpt_version = AudioSculpt_version.replace(' ', "\ ")# = os.path.normpath(AudioSculpt_version)
	return AudioSculpt_version