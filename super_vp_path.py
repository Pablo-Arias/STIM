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
	svp_version = glob.glob("/Applications/SuperVP/SuperVP.app/Contents/MacOS/*")[0]

	svp_version = svp_version.replace(' ', "\ ")# = os.path.normpath(AudioSculpt_version)
	return svp_version


if __name__ == "__main__":
	print(get_super_vp_path())