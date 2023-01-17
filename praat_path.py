def get_praat_path():
	import glob, os
	from sys import platform
	if platform == "linux" or platform == "linux2":
    	#Todo
		praat_path = "/home/pabloas/praat_nogui" #todo
		
	elif platform == "darwin":
    	# OS X
		praat_path = '/Applications/Praat.app/Contents/MacOS/Praat'
		

	return praat_path