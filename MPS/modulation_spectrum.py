# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ---------- Made by pablo Arias @ircam on 09/2016
# ---------- Copyright (c) 2018 CREAM Lab @ IRCAM
# ----------
# ---------- A set of python functions to compute the modulation spectrum
# ---------- these functions were intensiveley inspired from the Matlab 
# ---------- ModFilter Sound Tools
# ---------- Check ElliottTheunissen09 and Griffin 1984 for some papers
# ---------- The default parameters are the ones used in thuis toolbox
# ---------- 
# ---------- to us this don't forget to include these lines before your script:
# ----------
# ---------- import sys
# ---------- sys.path.append('/Users/arias/Documents/Developement/Python/')
# ---------- from modulation_spectrum import 'the functions you need'
# ----------
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#

from scikits.audiolab import wavread
import numpy as np
from scipy import signal
import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ---------- get_cmap_colors
def get_cmap_colors():
	#cmap for plotting spectrums
	colors = [(1.0000,    1.0000,    1.0000),
		(0.9524,    0.9952,    1.0000),
		(0.9048,    0.9848,    1.0000),
		(0.8571,    0.9686,    1.0000),
		(0.8095,    0.9467,    1.0000),
		(0.7619,    0.9190,    1.0000),
		(0.7143,    0.8857,    1.0000),
		(0.6667,    0.8467,    1.0000),
		(0.6190,    0.8019,    1.0000),
		(0.5714,    0.7514,    1.0000),
		(0.5238,    0.6952,    1.0000),
		(0.4762,    0.6333,    1.0000),
		(0.4286,    0.5657,    1.0000),
		(0.3810,    0.4924,    1.0000),
		(0.3333,    0.4133,    1.0000),
		(0.2857,    0.3286,    1.0000),
		(0.2381,    0.2381,    1.0000),
		(0.2390,    0.1905,    1.0000),
		(0.2457,    0.1429,    1.0000),
		(0.2581,    0.0952,    1.0000),
		(0.2762,    0.0476,    1.0000),
		(0.3000,         0,    1.0000),
		(     0,    0.5000,    0.0500),
		(     0,    0.5150,    0.0206),
		(0.0106,    0.5300,         0),
		(0.0436,    0.5450,         0),
		(0.0784,    0.5600,         0),
		(0.1150,    0.5750,         0),
		(0.1534,    0.5900,         0),
		(0.1936,    0.6050,         0),
		(0.2356,    0.6200,         0),
		(0.2794,    0.6350,         0),
		(0.3250,    0.6500,         0),
		(0.3724,    0.6650,         0),
		(0.4216,    0.6800,         0),
		(0.4726,    0.6950,         0),
		(0.5254,    0.7100,         0),
		(0.5800,    0.7250,         0),
		(0.6364,    0.7400,         0),
		(0.6946,    0.7550,         0),
		(0.7546,    0.7700,         0),
		(0.7850,    0.7536,         0),
		(0.8000,    0.7200,         0),
		(1.0000,    0.9000,    0.5000),
		(1.0000,    0.8766,    0.4750),
		(1.0000,    0.8515,    0.4500),
		(1.0000,    0.8246,    0.4250),
		(1.0000,    0.7960,    0.4000),
		(1.0000,    0.7656,    0.3750),
		(1.0000,    0.7335,    0.3500),
		(1.0000,    0.6996,    0.3250),
		(1.0000,    0.6640,    0.3000),
		(1.0000,    0.6266,    0.2750),
		(1.0000,    0.5875,    0.2500),
		(1.0000,    0.5466,    0.2250),
		(1.0000,    0.5040,    0.2000),
		(1.0000,    0.4596,    0.1750),
		(1.0000,    0.4135,    0.1500),
		(1.0000,    0.3656,    0.1250),
		(1.0000,    0.3160,    0.1000),
		(1.0000,    0.2646,    0.0750),
		(1.0000,    0.2115,    0.0500),
		(1.0000,    0.1566,    0.0250),
		(1.0000,    0.1000,         0)
		]
	return colors

# ---------- make_cmap
def make_cmap(colors, position=None, bit=False):
    """
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    """
    import matplotlib as mpl
    import numpy as np
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap

# ---------- compute_mod_spectrum
def compute_mod_spectrum(sound_in, fs, fband = 32, nstd = 6, DBNOISE = 60
						, logflg = True, plot_specgram = False
						, plot_fft2_before_shift       = False
						, plot_fft2_shifted            = False
						):
	"""
	This function computes the MPS (Modulation power spectrum) of a sound
 	
 	parameters:
		sound_in 				: data of sound in (only mono sounds please)
		fs 						: sampling frequency
		fband					: frequency band
		nstd					: number os standard deviations
		DBNOISE					: Noise floor : if (value of specgram - DBNOISE) < 0 ==> value = 0
		logflg					: Uee the log transform of the spectrogram before the FFt2
		plot_specgram			: If True, plots spectrogram of sound_in and paded silence
		plot_fft2_before_shift	: If true, plots the FFT2 transform of spectrogram
		plot_fft2_shifted		: If true, plots the FFT2 transform of spectrogram after fftshift


		This function returns :
		dwt 	 : temporal modulation axes of MPS (in Hz)
		dwf 	 : frequency modulation axes of MPS (in Cyc/kHz)
		amp_fabs : matrix of amplitude values of MPS
	
	warning : 
		this function only works for mono files
	"""
	twindow     = 1000*nstd/(fband*2.0*np.pi); 			# Window length in ms - nstd times the standard dev of the gaussian window
	window_size = np.fix(twindow*fs/1000.0);   			# Window length in number of points
	window_size = np.fix(window_size/2)*2;     			# Enforce even window length
	increment   = window_size - int(np.fix(0.001*fs));  # Sampling rate of spectrogram in number of points - set at 1 kHz
	ampsrate    = fs/increment;

	# Force sound_in to have zero mean.
	sound_in = sound_in - np.mean(sound_in);

	#window
	window = signal.gaussian(window_size, std=window_size/nstd, sym=False)

	#pad the sound with 0
	silence_time    = 0.5
	to_pad          = np.zeros(np.round(fs*silence_time))
	paded_sound_in  = np.concatenate([to_pad,sound_in,to_pad])

	#compute gaussian spectrogram
	f, t, Sxx = signal.spectrogram(paded_sound_in       , fs                 , nperseg = window_size 
	                               , nfft = window_size , noverlap=increment , window = window   
	                               , scaling ='spectrum', mode = 'magnitude'
	                              )

	
	#plot specgram
	if plot_specgram:
		plt.figure()
		fig, ax = plt.subplots()
		plt.pcolormesh(t, f, Sxx, cmap = 'nipy_spectral')
		ax.axis('tight')
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time [sec]')
		plt.title("Normal FFT of audio signal")
		plt.show()

	sabs = abs(Sxx)
	maxsabs = np.max(sabs)

	sabsDisp = 20*np.log10(sabs/maxsabs)+DBNOISE;
	sabsDisp[sabsDisp < 0] = 0

	if logflg:
		sabs = sabsDisp


	#number of time points and number of freq bins in spectrogram after pading
	[nb, nt] = sabs.shape
	fabs     = np.fft.fft2(sabs)
	amp_fabs = abs(fabs);
	amp_fabs = 20*np.log10(amp_fabs)

	
	#plot fft2 before freq shift
	if plot_fft2_before_shift:
		plt.figure()
		plt.pcolormesh(amp_fabs)
		plt.title("FFT2 before freq shift")

	#fft shift
	amp_fabs = np.fft.fftshift(amp_fabs)

	#plot shifted fft2
	if plot_fft2_shifted:
		plt.figure(figsize=(6, 4))
		plt.pcolormesh(amp_fabs)
		plt.title("FFT2 after freq shift")


	# Compute the labels for spectral modulations in physical units
	# f_step is the separation between frequency bands
	fstep    = f[1] - f[0];
	up_bound = int(np.ceil(nb))
	dwf      = np.zeros(up_bound)    
	dwf 	 = np.fft.fftfreq(int(np.ceil((nb+1))-1) , 1/(fstep*nb))
	dwf 	 = np.fft.fftshift(dwf)

	# Compute the labels for temporal modulations in physical units
	# ampsrate is the sampling rate of the amplitude enveloppe
	period    = t[len(t)-1]/len(t)
	up_bound  = int(np.ceil(nt))
	dwt       = np.zeros(up_bound)
	dwt       = np.fft.fftfreq(int(np.ceil((nt+1))-1) , period)
	dwt 	  = np.fft.fftshift(dwt)

	return dwt, dwf/1000, amp_fabs



# ---------- plot_MPS
def plot_MPS(amp_fabs, dwt, dwf, x_min = "" , x_max = "", y_min = "", y_max = ""
	, title = '', fig_name = '',interpolation= 'nearest', cmap = '', vmin = 0, vmax = 0):
	"""
	This function plots the MPS (Modulation power spectrum) of a sound 
	by using the parameters returned by the function compute_mod_spectrum
 	
 	parameters:
		amp_fabs    	: amp_fas as returned by compute_mod_spectrum
		dwt 			: dwt as returned by compute_mod_spectrum
		dwf 			: dwf as returned by compute_mod_spectrum
		x_min 			: min of x in graph when plotting
		x_max 			: max of x in graph when plotting
		y_min 			: min of y in graph when plotting
		y_max 			: max of y in graph when plotting
		title 			: title of plot
		fig_name 		:
		interpolation	: interpolation between the pixels, check matplotlib styles
		cmap 			: cmaps for heat, don't worry about this, I created a custom one for you
		vmin 			: minimum value to plot, interesting to erase noise
		vmax 			: maximum value to plot, interesting to erase noise

		This function returns :
			dwt 	 : temporal modulation axes of MPS (in Hz)
			dwf 	 : frequency modulation axes of MPS (in Cyc/kHz)
			amp_fabs : matrix of amplitude values of MPS
	
	"""	
	colors = get_cmap_colors()

	if vmin == 0 and  vmax == 0:
		vmin = np.min(amp_fabs)
		vmax = np.max(amp_fabs)

	
	plt.figure()
	im = plt.imshow(amp_fabs
	           , aspect='auto', extent = [min(dwt), max(dwt), min(dwf), max(dwf)]
	           , cmap = make_cmap(colors)
	           , interpolation = interpolation
	           , vmin = vmin
	           , vmax = vmax
	          );

	
	#fig axis
	if not (x_min == "" or x_max == "" or y_min == "" or y_max == "" ):
		plt.axis([x_min, x_max, y_min, y_max])

	plt.ylabel('Spectral modulations (Cycles/kHz)')
	plt.xlabel("Temporal Modulation (Hz)")
	
	#fig name
	if title == '':
		plt.title("Modulation Power Spectrum")
	else:
		plt.title(title)

	
	
	#save figure
	if fig_name != '':
		plt.savefig(fig_name)
		plt.close()
	else:
		plt.show()

#This function reduces the MPS matrix returned from the function compute_mod_spectrum
#to the ranges specified, this is done to zoom into the data easily
def take_reduced_matrix(amp_fabs, dwt, dwf, max_amp_mod  , min_amp_mod , max_freq_mod, min_freq_mod):
	"""
	This function reduces the MPS matrix returned from the function compute_mod_spectrum
	to the ranges specified, this is done to zoom into the data easily
 	
 	parameters:
 		amp_fabs  				: Matrix with the MPS data
 		dwt 					: temporal modulation axis 
 		dwf 					: frequency modulation axis
		max_amp_mod 			: maximum amplitude modulation
		min_amp_mod 			: minimum amplitude modulation
		max_freq_mod			: (Should be in Cycles/kHz as we do a conversion to Cycles/Hz inside th function, this is for ergonomicity of the function)
		min_freq_mod			: (Should be in Cycles/kHz as we do a conversion to Cycles/Hz inside th function, this is for ergonomicity of the function)

		This function returns :
		reduced_dwt 	 : temporal modulation axes of MPS (in Hz)
		reduced_dwf 	 : frequency modulation axes of MPS (in Cyc/Hz)
		reduced_amp_fabs : matrix of amplitude values of MPS
	
	warning : 
		Be careful dwt, max_amp_mod, min_amp_mod should have the same unit
		Be careul dwf, max_freq_mod, min_freq_mod should have the same unit
	"""
	from bisect import bisect_left
	import numpy as np

	#transform freq mod to cyc/Hz
	max_amp_mod_ind  = bisect_left(dwt, max_amp_mod)
	min_amp_mod_ind  = bisect_left(dwt, min_amp_mod)
	max_freq_mod_ind = bisect_left(dwf, max_freq_mod)
	min_freq_mod_ind = bisect_left(dwf, min_freq_mod)

	#take reduced matrix and axis
	reduced_matrix = amp_fabs[min_freq_mod_ind:max_freq_mod_ind, min_amp_mod_ind:max_amp_mod_ind]
	reduced_matrix = np.flipud(reduced_matrix)
	reduced_dwt    = dwt[min_amp_mod_ind : max_amp_mod_ind]   # in Cycles/kHz
	reduced_dwf    = dwf[min_freq_mod_ind : max_freq_mod_ind] # in Cycles/kHz

	return reduced_matrix, reduced_dwt, reduced_dwf   

