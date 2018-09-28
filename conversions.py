#-*- coding: utf-8 -*-
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ---------- Made by pablo Arias @ircam on 11/2015
# ---------- Copyright (c) 2018 CREAM Lab // CNRS / IRCAM / Sorbonne Université
# ----------
# ---------- make data conversions 
# ----------
# ---------- import sys
# ---------- sys.path.append('/Users/arias/Documents/Developement/Python/')
# ---------- from conversion import 'the functions you need'
# ----------
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#

import numpy as np


def lin2db(vec): return 20.*np.log10(vec)
def db2lin(vec): return 10**(vec/20.)