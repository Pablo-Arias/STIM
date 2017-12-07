# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ---------- Made by pablo Arias @ircam on 09/2016
# ---------- This script plot the MPS in real time, it captures the sound for the 
# ---------- system input (Change this one to be soundflowerbed for example and 
# ---------- the output of live to be soundflowerbed)
# ---------- 
# ----------
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
#imports
import pyaudio
import scipy.fftpack
import scipy.interpolate
import numpy
import pylab
import numpy as np
import matplotlib.pyplot as plt

#Local
import sys
sys.path.append('/Users/arias/Documents/Developement/Python/')
from modulation_spectrum import compute_mod_spectrum, plot_MPS,take_reduced_matrix, get_cmap_colors

def make_cmap(colors, position=None, bit=False):
    '''
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
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

# ------------------------------------------------
# ------------------------------------------------
# ------------------------------------------------
# ------------- Begin Script ---------------------
# ------------- Real Time MPS computing ----------
# ------------- and rendering --------------------
# ------------------------------------------------
# ------------------------------------------------
# ------------------------------------------------

#Modulation Spectrum Init - Change this if needed

#Init - MPS params 
Threshold_amp_mod 	= 90 

#Init - plot params
v_min 				= 60   # Min value of intensity bin to plot
v_max 				= 140  # Max value of intensity bin to plot
x_min 				= -40  # Abscisse min in the plot
x_max 				= 40   # Abscisse max in the plot
y_min 				= -10  # Ordinate min in the plot
y_max 				= 10   # Ordinate max in the plot


#Init - pyAudio settings
settings = {
	"format": pyaudio.paInt16,
	"channels": 1,
	"rate": 96000,
	"input": True,
	"frames_per_buffer": 4096*4
}
audio  = pyaudio.PyAudio()
print "Chosen device : " + audio.get_default_input_device_info().get('name')
stream = audio.open(**settings)


#pyLab configurations
#pylab.ion()
#figura, (spectrum_log, spectrum, wave) = pylab.subplots(nrows=3, ncols=1)
#pylab.subplots_adjust(hspace=0.5, left=0.1)

fig, ax = plt.subplots()
block = np.zeros(settings.get("frames_per_buffer"))


dwt, dwf, amp_fabs = compute_mod_spectrum(block, settings.get("rate"), fband = 32, nstd = 6, DBNOISE = 500
                                , logflg                  = True
                                , plot_specgram           = False
                                , plot_fft2_before_shift  = False
                                , plot_fft2_shifted       = False
                            )

#Plot
lines =  plt.imshow(np.random.rand(*amp_fabs.shape)
			, aspect='auto', extent = [x_min, x_max, y_min, y_max]
			, cmap = make_cmap(get_cmap_colors())
			, interpolation = 'gaussian'
			, vmin = v_min
			, vmax = v_max
			);

plt.colorbar(lines)
fig.canvas.manager.show() 

# #Main loop
while True:
	try:
		#Acquire data from microphone
		block = numpy.array(numpy.fromstring(
							stream.read(settings["frames_per_buffer"]), dtype="int16")
						)

		#Compute modulation Spectrum
		dwt, dwf, amp_fabs = compute_mod_spectrum(block, settings.get('rate'), fband = 32, nstd = 6, DBNOISE = 500
		                                , logflg                  = True
		                                , plot_specgram           = False
		                                , plot_fft2_before_shift  = False
		                                , plot_fft2_shifted       = False
		                            )
		
		#Update plot
		lines.set_data(amp_fabs)
		fig.canvas.draw()
		fig.canvas.flush_events()

	except IOError:
		continue