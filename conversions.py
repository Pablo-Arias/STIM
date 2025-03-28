#-*- coding: utf-8 -*-
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ---------- Made by pablo Arias @ircam on 11/2015
# ---------- Copyright (c) 2018 CREAM Lab // CNRS / IRCAM / Sorbonne Universite
# ----------
# ---------- useful file/data/metric conversions 
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#

import numpy as np
import os

def lin2db(vec):
    epsilon = 1e-10  # Small constant to avoid log10(0)
    return 20. * np.log10(np.maximum(vec, epsilon))

def db2lin(vec): return 10**(vec/20.)

def hz_to_cents(f0, f1):
	"""
	function to convert the interval [f0, f1]  in Hz to cents
	"""
	if np.isnan(f0) or np.isnan(f1):
		return np.nan

	return 1200* np.log2(f1/float(f0))


#name conversions
def get_file_without_path(file_name, with_extension=False):
	"""
		get the name of a file without its path
	"""

	base = os.path.basename(file_name)
	if not with_extension:
		base = os.path.splitext(base)[0]
	
	return base

def get_file_path(file_name):
	return "/".join(file.split("/")[:-1])