#-*- coding: utf-8 -*-
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ---------- Made by pablo Arias @ircam on 11/2015
# ---------- Copyright (c) 2018 CREAM Lab // CNRS / IRCAM / Sorbonne Universite
# ----------
# ---------- make useful file/data/metric conversions 
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#

import numpy as np
import os

def lin2db(vec): return 20.*np.log10(vec)
def db2lin(vec): return 10**(vec/20.)

#name conversions
def get_file_without_path(file_name, with_extension=False):
	"""
		get the name of a file without its path
	"""

	base = os.path.basename(file_name)
	if not with_extension:
		base = os.path.splitext(base)[0]
	
	return base