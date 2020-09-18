#-*- coding: utf-8 -*-
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ---------- Made by pablo Arias @ircam on 11/2015
# ---------- Copyright (c) 2018 CREAM Lab / CNRS / IRCAM / Sorbonne Universite
# ----------
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
from __future__ import absolute_import
def get_super_vp_path():
	import glob, os
	AudioSculpt_version = glob.glob("/Applications/AudioSculpt*/AudioSculpt*.app/Contents/MacOS/supervp")[0]
	AudioSculpt_version = AudioSculpt_version.replace(' ', "\ ")# = os.path.normpath(AudioSculpt_version)
	return AudioSculpt_version