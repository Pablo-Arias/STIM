from pyo import *
import glob
from scipy.io.wavfile import read, write
from scikits.audiolab import aiffread, wavwrite  # scipy.io.wavfile can't read 24-bit WAV
import aifc
import numpy as np
from numpy import pi, polymul
import os

def Lin2db(Lin):
	return 20*np.log10(Lin)

def rms_flat(a):  # from matplotlib.mlab
	"""
	Return the root mean square of all the elements of *a*, flattened out.
	"""
	return np.sqrt(np.mean(np.absolute(a)**2))

def IndexFileInFolder(FolderName):
	for file in glob.glob(FolderName + "/*.wav"): # Wav Files
		x, fs, enc 		= aiffread(str(file))
		WndSize 		= 16384
		rmsTreshhold 	= -50 

		index = 0
		NbofWrittendFiles = 1
		while index + WndSize < len(x):
	 		DataArray 	= x[index: index + WndSize]
	 		rms 		= np.sqrt(np.mean(np.absolute(DataArray)**2))
	 		rms 		= Lin2db(rms)
			index 		= WndSize + index
			if rms > -55:
				end 		= 0
				begining 	= index
				index 		= WndSize + index
				while rms > -55:
					if index + WndSize < len(x):
						index 		= WndSize + index
						DataArray 	= x[index: index + WndSize]
	 					rms 		= np.sqrt(np.mean(np.absolute(DataArray)**2))
	 					rms 		= Lin2db(rms)
	 					end 		= index
	 				else:
	 					break

				
	 			#if file is over 500 ms long, write it
	 			if (end - begining) > (fs / 2) :
	 				duree = (end - begining) / float(fs)
	 				print  "duree  :  "+ str(duree)
	 				
	 				begining = begining - WndSize
	 				if begining < 0: 
	 					begining = 0
	 				 
	 				end = end + WndSize
	 				if end > len(x): 
	 					end = len(x)

	 				name = os.path.splitext(str(file))[0]
	 				name  = os.path.basename(name)
	 				wavwrite(x[begining:end],"Indexed/"+"/"+ FolderName +"/"+ name + "_" +str(NbofWrittendFiles)+ ".wav", fs, enc='pcm24')
	 				NbofWrittendFiles = NbofWrittendFiles + 1


folders = ('F2', 'F3')
for FolderName in folders:
#	FolderName = str(x[0])
#	FolderName = FolderName[2:len(FolderName)]
	
	print FolderName
	IndexFileInFolder(FolderName)








