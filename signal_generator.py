# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ---------- Made by Pablo Arias @ircam on 11/2015
# ---------- plot sound features
# ---------- to us this don't forget to include these lines before your script:
# ----------
# ---------- import sys
# ---------- sys.path.append('/Users/arias/Documents/Developement/Python/')
# ---------- from signal_generator import 'the functions you need'
# ----------
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#


from pylab import plot, show, title, xlabel, ylabel, subplot, savefig, grid, specgram, suptitle, yscale
from scipy import fft, arange, ifft, io
from numpy import sin, linspace, pi, cos, sin, random, array
import numpy as np
from scipy.io.wavfile import read, write
from matplotlib import pyplot
from scikits.audiolab import wavwrite

def Normalize(data):
	Max  = max(data)
   	data = [x / Max for x in data]
   	return data

def WhiteNoiseGenerator(Duration, Fs):
	t    = linspace(0, Duration, Duration * Fs, endpoint=True)
	data = []
	data = random.normal(0, 0.5, Duration*Fs)
	#data = Normalize(data)
	data = np.array(data, dtype=np.float32)
	return data

def Ramp(freq, Duration, Fs):
	Data = []
	Ramp = []
	DureeRamp = float(1)/freq
	Ramp = linspace(0, DureeRamp , DureeRamp*Fs, endpoint=True)
 	while len(Data) < Duration*Fs:
		Data.extend(Ramp)	
	Data = Data[0:Duration*Fs]	
	Data = Normalize(Data)
	return Data

def Triangle(freq, Duration, Fs):
	Data, Ramp = [], []
	DureeRamp = float(1)/freq
	UpRamp 	 = linspace(0, DureeRamp , (DureeRamp*Fs/2))
 	DownRamp = linspace(DureeRamp, 0 , (DureeRamp*Fs/2))
	
 	Ramp.extend(UpRamp)
	Ramp.extend(DownRamp)

 	while len(Data) < Duration*Fs:
		Data.extend(Ramp)	

	Data = Data[0:Duration*Fs]	
	Data = Normalize(Data)
	return Data

def Sin(freq, Duration, Fs):
	t    = linspace(0, Duration , Duration * Fs, endpoint=True)
	data = sin(2*pi*freq*t)
	return data

def Cos(freq, Duration, Fs):
	t    = linspace(0, Duration , Duration * Fs, endpoint=True)
	data = cos(2*pi*freq*t)
	return data

def SquareWave(freq, Duration, Fs):
	t    = linspace(0, Duration, Duration * Fs, endpoint=True)
	Data = []
	SamplesPerPeriod = Fs/ freq 
	for i in range(0, len(t)):
		if (i % SamplesPerPeriod) < SamplesPerPeriod/2 :
			Data.append(1.0)
		else :
			Data.append(0.0)
	return Data

def pink_noise_to_file(duration, filename, mono = True, sampletype = 1):
	"""
	filename : file to generate
	duration : duration in seconds of the file to generate
	fs       : fs of the file to generate
	sampletype : 16 bits int (default)
				 24 bits int
				 32 bits int
				 32 bits float
				 64 bits float
				 U-Law encoded
				 A-Law encoded
	
	mono : generates mono file if mono = True
	"""
	from pyo import Server, PinkNoise

	s = Server(duplex=0, audio="offline").boot()
	s.boot()
	name = filename
	s.recordOptions(dur = duration + 0.1, filename=filename, fileformat=0, sampletype=sampletype)
	a = PinkNoise(.1).out()
	s.start()
	s.shutdown()

	if mono == True:
		from transform_audio import wav_to_mono
		wav_to_mono(filename, filename)


def generate_silence_sound(duration, fs, name, enc = "pcm16"):
	"""
	duration : in seconds
	fs       : sampling frequency
	name     : file name to generate
	enc      :  pcm16, pcm24 ...
	"""

	import numpy as np
	from scikits.audiolab import wavwrite
	data = np.zeros(duration*fs)
	wavwrite(data, name, fs, enc)
