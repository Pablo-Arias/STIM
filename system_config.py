# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ---------- Made by pablo Arias @ircam on 11/2015
# ---------- Copyright (c) 2018 CREAM Lab // CNRS / IRCAM / Sorbonne Université
# ----------
# ---------- functions for system configuration
# ---------- 
# ----------
# ---------- import sys
# ---------- sys.path.append('/Users/arias/Documents/Developement/Python/')
# ---------- from system_config import 'the functions you need'
# ----------
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#

import pip

def print_installed_packages():
	installed_packages 		= pip.get_installed_distributions()
	installed_packages_list = sorted(["%s==%s" % (i.key, i.version)

	for i in installed_packages])
		print(installed_packages_list)