#-*- coding: utf-8 -*-
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ---------- Made by pablo Arias @ircam on 11/2015
# ---------- Copyright (c) 2018 CREAM Lab // CNRS / IRCAM / Sorbonne Université
# ----------
# ---------- functions for calling praat scripts from python 
# ----------
# ---------- import sys
# ---------- sys.path.append('/Users/arias/Documents/Developement/Python/')
# ---------- from praat_scripts import 'the functions you need'
# ----------
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#

import numpy as np
import subprocess

#TODO : this should work but putting it inside a function seem to not work....
def Extract_formants_with_praat(audio_file):
	x = subprocess.check_output(['/Applications/Praat.app/Contents/MacOS/Praat','extract_formants.praat', audio_file]);
	F1, F2, F3 = x.splitlines()
	return F1, F2, F3


