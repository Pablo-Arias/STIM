#-*- coding: utf-8 -*-
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ---------- Made by Pablo Arias @ircam on 11/2015
# ---------- Copyright (c) 2018 CREAM Lab // CNRS / IRCAM / Sorbonne Université
# ---------- plot sound features
# ---------- to us this don't forget to include these lines before your script:
# ----------
# ---------- import sys
# ---------- sys.path.append('/Users/arias/Documents/Developement/Python/')
# ---------- from plot_features import 'the functions you need'
# ----------
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#

from pylab import plot, show, title, xlabel, ylabel, subplot, savefig, grid, specgram, suptitle, yscale
from scipy import fft, arange, ifft, io
from numpy import sin, linspace, pi, cos, sin, random, array
import numpy as np
from scipy.io.wavfile import read, write
from matplotlib import pyplot
import matplotlib.pyplot as plt
from pylab import *
import os
from scikits.audiolab import wavread

# ---------- local imports
from parse_sdif import mean_matrix
from super_vp_commands import generate_LPC_analysis, generate_true_env_analysis, generate_fft_analysis
from audio_analysis import mean_formant_from_audio
from audio_analysis import formant_dataframe_to_arrays

#plot_colors = ["red", "red", "green", "blue"]
plot_colors = ["black", "black", "black", "grey", "yellow", "black", "black"]
# ---------- plot_cos_and_sin
def plot_cos_and_sin():
    X 	 = linspace(-pi, pi, 256, endpoint=True)
    C, S = cos(X), sin(X)
    plot(X, C)
    plot(X, S)
    show()

# ---------- PlotPSForWavFile
def Plot_PS_for_wav_file(name, Duration):
	"""
	Plot Power Spectrum of wave File
	"""
	rate, data = io.wavfile.read(name)
	Data = data[0:Duration*rate]
	plot_PS(data, rate)


# ---------- plot_PS
def plot_PS(data, Fs):
	"""
	Plot Power Spectrum of data
	"""
	y 	 = data
	timp = len(y)/Fs
	t 	 = linspace(0,timp,len(y))

	pyplot.subplot(2,1,1)
	pyplot.plot(t,y)
	pyplot.xlabel('Time')
	pyplot.ylabel('Amplitude')
	pyplot.subplot(2,1,2)

	n 	= len(y)
	k 	= arange(n)
	T 	= n/Fs
	frq = k/T # two sides frequency range
	frq = frq[range(n/2)] # one side frequency range
 	Y 	= fft(y)/n # fft computing and normalization
 	Y 	= Y[range(n/2)]
 	
	pyplot.xscale('linear')
	pyplot.plot(frq, abs(Y),'r') # plotting the spectrum
	pyplot.xlabel('Freq (Hz)')
	pyplot.ylabel('|Y(f)|')

	show()

# ---------- plot_spec_gram_from_audio
def plot_spec_gram_from_audio(FileName, NFFT=512):
	import matplotlib.pyplot as plt
	sound_data, fs, enc = wavread(FileName)	

	#Spectre
	plt.style.use('ggplot')
	xlabel('Time (s)')
	ylabel('|Y(freq)|')
	suptitle('Spectrogram', fontsize=15)
	grid()
	Pxx, freqs, bins, im = plt.specgram(sound_data, NFFT=NFFT, noverlap=128 , Fs=fs)
	plt.show()
	return Pxx, freqs, bins, im

# ---------- plot_spec_gram
def plot_spec_gram(rate, data):	
	Pxx, freqs, bins, im = specgram(data, NFFT=512, Fs=rate, noverlap=10)
	xlabel('Time (s)')
	ylabel('|Y(freq)|')
	suptitle('Spectrogram', fontsize=15)
	yscale('symlog')
	grid()
	show()

# --------------------------------------------
# ----------- plot spectral envelope
# --------------------------------------------

def get_f(nb_points):
	return np.arange(20,22000, (22000.0 - 20)/nb_points) 

def spectral_env_plot_params(ax, x_limit = (500, 20000), envelope = None, adaptive_x_grid = True, grid = False, margin = 5, x_log = False, plot_legend = True):
	ax.grid(grid)

	if x_log:
		ax.set_xscale('log')

	if plot_legend:
		ax.set_xlabel('Frequency (Hz)', fontsize=30)
		ax.set_ylabel('Amplitude (db)', fontsize=30)
		legend(loc='best', prop={'size':26})
	
	else:
		plt.tick_params(axis='both'
				, which='both'
				, bottom='off'
				, top='off'
				, labelbottom='off'
				, right='off'
				, left='off'
				, labelleft='off'
				)
		
	
	
	xlim(x_limit[0], x_limit[1])

	if adaptive_x_grid:
		ylim(np.min(envelope) - margin, np.max(envelope) + margin)

	


def plot_spectrogram(audio_file, window_size = 2048, nfft = 256 , noverlap = 255 
					, plot_legends = True, interpolation = 'nearest'
					, x_min = "" , x_max = "", y_min = "", y_max = ""
					, noise = 0.01
					, save_pdf_with_name = ""
					, title = ""
					, xlog = False
					):
	"""
	parameters:
		audio_file 			: path to audio file to plot
		window_size  		: window size to use
		nfft				: nfft
		noverlap			: number of samples of overlap
		plot_legends		: if True plot the axis legends
		interpolation		: matplotlib's interpolation
		x_min				: min time value to plot 
		x_max				: max time value to plot
		y_min				: min freq value to plot
		x_max				: max freq value to plot
		noise				: noise to remove from the ploting

	keep in mind : 
		This only works with mono files
	"""	

	from scikits.audiolab import wavread
	from scipy import signal
	from MPS.modulation_spectrum import make_cmap, get_cmap_colors
	from matplotlib.colors import LogNorm


	# Force sound_in to have zero mean.
	sound_in, fs, enc = wavread(audio_file)

	#compute gaussian spectrogram
	f, t, Sxx = signal.spectrogram(sound_in, fs, nperseg = window_size 
	                               , nfft = window_size , noverlap=noverlap 
	                               , scaling ='spectrum', mode = 'magnitude'
	                              )
		
	#plot specgram	
	colors = get_cmap_colors()
	plt.figure()

	plt.imshow(Sxx
				, aspect ='auto'
				, extent = [min(t), max(t), min(f), max(f)]
				#, cmap = 'Greys'#make_cmap(colors) #'nipy_spectral'#'Greys' #make_cmap(colors)
				, cmap = make_cmap(colors) #'nipy_spectral'#'Greys'
				, interpolation = interpolation
				, origin = 'lower'
				, vmin = np.min(Sxx) + noise
				, vmax = np.max(Sxx)
				, norm=LogNorm()
	          )
	#plt.axis("off")
	#fig axis
	if not (x_min == "" or x_max == "" or y_min == "" or y_max == "" ):
		plt.axis([x_min, x_max, y_min, y_max])

	#ax.axis('tight')
	
	if plot_legends:
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time [sec]')

		if title != "":
			plt.title(title)
	else:
		plt.tick_params(axis='both'
						, which='both'
						, bottom='off'
						, top='off'
						, labelbottom='off'
						, right='off'
						, left='off'
						, labelleft='off'
						)
	if save_pdf_with_name != "":
		plt.savefig(save_pdf_with_name)

	plt.show()


# ---------- plot_spectral_envelope_from_data
def plot_spectral_envelope_from_data(datas, label = None, show = True, save_pdf_with_name = "", title='', figsize=(25, 8), dpi=80, lw=2, x_log = False, plot_legend = True, x_limit = (0,22000)):
	"""
	parameters:
		data 						: Array with data to plot, each item in datas will be ploted as a separate curve
		label  						: Array with curve label
		show						: show the plot
		save_pdf_with_name			: save_pdf with the name in argument. the .pdf extension is not needed, if not defined no pdf will be created.
		title						: Title of your figure
		figsize						: figure size
		dpi 						: Pixel density 
		lw 							: Line width
		x_limit 						: limits of x
	keep in mind : 
		This only works with mono files
	"""	
	fig 		= plt.figure(figsize=figsize, dpi=dpi)

	ax 			= fig.add_subplot(1,1,1)

	plt.rc('xtick', labelsize = 15) 
	plt.rc('ytick', labelsize = 15) 
	style.use('seaborn-white')


	for i in range(len(datas)):
		data = datas[i]
		nb_points 	= len(data)
		f 			= get_f(nb_points)
		
		#from conversions import lin2db
		#data = lin2db(data)
		linestyles = ['-', '--', ':', '-', '--', '--', '--']

		if label == None or not plot_legend:
			line, = ax.plot(f, data, linestyles[i],lw=lw, color = plot_colors[i], )
			#line, = ax.plot(f, data,lw=lw, color = plot_colors[i])
		else:
			line, = ax.plot(f, data, linestyles[i],lw=lw, label = label[i], color = plot_colors[i])
			#line, = ax.plot(f, data,lw=lw, label = label[i], color = plot_colors[i])

	spectral_env_plot_params(ax, envelope = data, x_log = x_log, plot_legend = plot_legend, x_limit = x_limit)
	
	ax.grid(b=False)

	
	if plot_legend:
		plt.title(title, y=1.02, fontsize = 27)

	else:
		ax.set_xticklabels(["", "", ""])
		ax.set_yticklabels(["", "", ""])
		plt.title("", y=1.02, fontsize = 10)
		#plt.axis('off')
		
	
	plt.draw()
	

	if save_pdf_with_name != "":
		savefig( save_pdf_with_name + '.pdf', transparent=True)

	if show:
		plt.show()
	

# ---------- plot_mean_matrix_from_sdif
def plot_mean_matrix_from_sdif(sdifs, label = None, show = True, save_pdf_with_name = "", title='', figsize=(25, 8), dpi=80, lw=2, x_log = False, x_limit= (0,22000), plot_legend = True):
	"""
	parameters:
		sdifs 						: Array with files to plot
		label  						: Array with curve label
		show						: show the plot
		save_pdf_with_name			: save_pdf with the name in argument. the .pdf extension is not needed, if not defined no pdf will be created.
		x_limit                     : tuple with min and max x values

	keep in mind : 
		This only works with mono files

	"""	
	datas = []
	for i in range(len(sdifs)):	
		datas.append(mean_matrix(sdifs[i]))
	
	plot_spectral_envelope_from_data(datas, label = label, show = show, save_pdf_with_name = save_pdf_with_name, title=title, figsize=figsize, dpi=dpi, lw=lw, x_log = x_log, x_limit = x_limit ,plot_legend= plot_legend)

	


# ---------- plot_mean_lpc_from_audio
def plot_mean_lpc_from_audio(audio_files, label = None, show = True, save_pdf_with_name = ""
							, destroy_analysis_after_plot= False, title='', figsize=(20, 10), dpi=80, nb_coefs = 1024, lw=2
							, x_log = False
							, plot_legend = True
							, x_limit = (0,22000)
							):
	"""
	parameters:
		audio_files 				: Array with files to plot
		label  						: curve label
		show						: show the plot
		save_pdf_with_name			: save_pdf with the name in argument. the .pdf extension is not needed.
		destroy_analysis_after_plot : if set to True the analysis file created will be deleted after closing the plot window.
		nb_coefs					: Number of coeficients for the lpc analysis
		lw 							: Line width
		x_log 						: Use log in the x scale

	keep in mind : 
		This function creates an sdif data analysis with the same name and in the same path of audio_file by using super vp
		This only works with mono files
	"""
	#first do the Lpc analysis using supervp
	sdifs = []
	for audio_file in audio_files:
		
		file_tag = os.path.basename(audio_file)
		file_tag = os.path.splitext(file_tag)[0]
		analysis = os.path.dirname(audio_file)+ "/" +file_tag + ".sdif"

		sdifs.append(analysis)
		generate_LPC_analysis(audio_file, analysis, wait =  True, nb_coefs = nb_coefs)

	plot_mean_matrix_from_sdif(sdifs, label = label, show = show
						, save_pdf_with_name = save_pdf_with_name, title = title, figsize = figsize
						, dpi = dpi, lw = lw, x_log = x_log, plot_legend= plot_legend
						, x_limit               = x_limit						
						)
	
	if destroy_analysis_after_plot:
		for sdif in sdifs:
			os.remove(sdif)	


# ---------- plot_mean_true_env_from_audio
def plot_mean_true_env_from_audio(audio_files, label = None, show = True, save_pdf_with_name = "", destroy_analysis_after_plot= False
								, title='', figsize=(20, 10), dpi=80, lw=2, f0 = 100
								, x_log = False
								, x_limit = (0,22000)
								, plot_legend = True
								):
	"""
	parameters:
		audio_files 				: Array with files to plot
		label  						: curve label
		show						: show the plot
		save_pdf_with_name			: save_pdf with the name in argument. the .pdf extension is not needed.
		destroy_analysis_after_plot : if set to True the analysis file created will be deleted after closing the plot window.
		f0							: mean f0 of file this improves the analysis
		x_limit                     : limits of x axis

	keep in mind : 
		This function creates an sdif data analysis with the same name and in the same path of audio_file by using super vp
		This only works with mono files
	"""
	#first do the Lpc analysis using supervp
	sdifs = []
	for audio_file in audio_files:
		
		file_tag = os.path.basename(audio_file)
		file_tag = os.path.splitext(file_tag)[0]
		
		if os.path.dirname(audio_file) == "":
			analysis =  file_tag + ".sdif"
		else:
			analysis =  os.path.dirname(audio_file) + "/" + file_tag + ".sdif"

		sdifs.append(analysis)
		generate_true_env_analysis(audio_file, analysis, wait =  True)


	plot_mean_matrix_from_sdif(sdifs
									, label 				= label
									, show 					= show
									, save_pdf_with_name 	= save_pdf_with_name 
									, title 				= title
									, figsize 				= figsize
									, dpi 					= dpi
									, lw 					= lw
									, x_log 				= x_log
									, x_limit               = x_limit
									, plot_legend 			= plot_legend
								)
	
	if destroy_analysis_after_plot:
	 	for sdif in sdifs:
	 		os.remove(sdif)



# ---------- plot_mean_spectrum_from_audio
def plot_mean_spectrum_from_audio(audio_files, label = None, show = True, save_pdf_with_name = ""
								, destroy_analysis_after_plot= False, title='', figsize=(20, 10), dpi=80, lw=2
								, x_log = False
								, plot_legend = True
								):
	"""
	parameters:
		audio_files 				: Array with files to plot
		label  						: curve label
		show						: show the plot
		save_pdf_with_name			: save_pdf with the name in argument. the .pdf extension is not needed.
		destroy_analysis_after_plot : if set to True the analysis file created will be deleted after closing the plot window.

	keep in mind : 
		This function creates an sdif data analysis with the same name and in the same path of audio_file by using super vp
		This only works with mono files
	"""
	#first do the Lpc analysis using supervp
	sdifs = []
	for audio_file in audio_files:
		
		file_tag = os.path.basename(audio_file)
		file_tag = os.path.splitext(file_tag)[0]
		
		if os.path.dirname(audio_file) == "":
			analysis =  file_tag + ".sdif"
		else:
			analysis =  os.path.dirname(audio_file) + "/" + file_tag + ".sdif"

		sdifs.append(analysis)
		generate_fft_analysis(audio_file, analysis = "", wait = True)


	plot_mean_matrix_from_sdif(sdifs
								, label =  label
								, show =  show
								, save_pdf_with_name =  save_pdf_with_name
								, title =  title
								, figsize = figsize
								, dpi = dpi
								, lw =  lw
								, x_log = x_log
								, plot_legend = plot_legend
								)
	
	if destroy_analysis_after_plot:
	 	for sdif in sdifs:
	 		os.remove(sdif)	 		

# ---------- freqs_to_bins	 					
def freqs_to_bins(freq_array, nb_points, min_freq, max_freq):
	"""
	parameters:
		bins 						: Array with bins to transform
		nb_points					: total nb points of of array containing 
		max_freq					: max frequency represented by nb_points

	"""
	bins = []
	for freq in freq_array:
		bin_value = freq*nb_points/(max_freq- min_freq)
		bins.append(bin_value)

	return bins

# ---------- plot_formants_on_true_envelope
def plot_formants_on_true_envelope(audio_files, label = None, show = True, save_pdf_with_name = ""
								, destroy_analysis_after_plot= False, title='', figsize=(20, 10)
								, dpi=80, lw=2, f0 = 100, nb_formants = 5, use_t_env = False):
	"""
	parameters:
		audio_files 				: Array with files to plot
		label  						: curve label
		show						: show the plot
		save_pdf_with_name			: save_pdf with the name in argument. the .pdf extension is not needed.
		destroy_analysis_after_plot : if set to True the analysis file created will be deleted after closing the plot window.
		f0							: mean f0 of file, this improves the analysis
		nb_formants 				: n_formants to estimate
		use_t_env 					: Use true envelope analysis for formant extraction 

	keep in mind : 
		This function creates an sdif data analysis with the same name and in the same path of audio_file by using super vp
		This only works with mono files
	"""
	#first do the Lpc analysis using supervp
	sdifs = []
	for audio_file in audio_files:
		file_tag = os.path.basename(audio_file)
		file_tag = os.path.splitext(file_tag)[0]
		if os.path.dirname(audio_file) == "":
			analysis =  file_tag + "_true_env.sdif"
		else:
			analysis =  os.path.dirname(audio_file) + "/" + file_tag + "_true_env.sdif"

		sdifs.append(analysis)
		generate_true_env_analysis(audio_file, analysis, wait =  True)

	datas = []
	for i in range(len(sdifs)):	
		datas.append(mean_matrix(sdifs[i]))

	plt.style.use('ggplot')
	fig 		= plt.figure(figsize=figsize, dpi=dpi)
	ax 			= fig.add_subplot(1,1,1)
	
	for i in range(len(datas)):
		data = datas[i]
		nb_points 	= len(data)
		f 			= get_f(nb_points)
		if label == None:
			line, = ax.plot(f, data, lw=lw, color = plot_colors[i])
		else:
			line, = ax.plot(f, data, lw=lw, label = label[i], color = plot_colors[i])

		# Draw formants
		mean_formant = []
		bins 		 = []
		mean_formants = mean_formant_from_audio(audio_files[i], nb_formants, use_t_env = use_t_env)

		frequency, amplitude, bandwith, Saliance = formant_dataframe_to_arrays(mean_formants)
		bins = freqs_to_bins(frequency, nb_points, min_freq = 20, max_freq = 22000)
		if len(bins) > 0 :
			for index in range(len(bins)):
				plt.plot(frequency[index], [data[bins[index]]], 'ro', color = plot_colors[index])

	spectral_env_plot_params(ax)
	plt.draw()
	plt.title(title)

	if save_pdf_with_name != "":
		savefig( save_pdf_with_name + '.pdf')

	if show:
		plt.show()

	
	if destroy_analysis_after_plot:
	 	for sdif in sdifs:
	 		os.remove(sdif)			

# ---------- plot_formant_parameter
def plot_formant_parameter(audio_file, nb_formants = 5, parameter = "", show = True ,save_pdf_with_name = "",figsize=(20, 10), dpi=80, lw=2):
	"""
	audio_file : file to analyse and plot features
	parameter to plot can be:
		"Frequency", "Amplitude", "Bw", "Saliance"
	"""
	from audio_analysis import formant_from_audio
	formants  = formant_from_audio(audio_file, nb_formants)	
	frequency, amplitude, bandwith, Saliance  = formant_dataframe_to_arrays(formants)

	print frequency
	time = frequency[0].index

	if parameter == "Frequency":
		datas = frequency

	if parameter == "Amplitude":
		datas = amplitude
	if parameter == "Bw":
		datas = bandwith
	if parameter == "Saliance":
		datas = Saliance

	labels = []
	for i in range(1 , nb_formants + 1):
		labels.append(parameter + str(i))

	plt.style.use('ggplot')
	fig 		= plt.figure(figsize=figsize, dpi=dpi)
	ax 			= fig.add_subplot(1,1,1)

	xlabel('Time (s)')
	ylabel('|Y(freq)|')
	suptitle(parameter + " over time ", fontsize=15)
	yscale('symlog')
	grid()


	for i in range(len(datas)):
		data = datas[i].values
		line, = ax.plot(time, data, label = labels , lw=lw) 


	if save_pdf_with_name != "":
		savefig( save_pdf_with_name + '.pdf')

	if show:
		plt.show()



