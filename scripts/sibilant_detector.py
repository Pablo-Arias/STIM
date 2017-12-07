
from scikits.audiolab import wavread
from scikits.audiolab import wavwrite
import operator

def weighting_vector(size, sigma, mu, fs):
    weights = []
    mu = mu/(fs/2) * size
    weights = []

    for t in range(size):
        a = 1/(sigma * np.sqrt(2 * math.pi *sigma))
        val = a * math.exp( - math.pow(t - mu, 2) / (2 * math.pow(sigma, 2)))
        weights.append(val)

    return weights



def sibilant_detector(filename):
	"""
	The aim of this algorithm is to detect where are the parts in filename where the energy is maximal.
	This algorithm works as follows:
	1- First compute the spectrogram
	2- Then compute a gaussian curve centered in the frequency researched. Usually for sibilants it's around 6000 Hz
	3- Multiply the spectrum and the gaussian in order to weight the spectrum
	4- Mean all the resultant signal and normalize
	5- The peaks in the resulting signal are the parts in time where the energy in the researched area is the most important.
	"""
	sound_data, fs, enc = wavread(filename)
	
	#Gaussian coefs
	sigma = 5
	mu = 10000 # mean frequency
	NFFT=512

	#Spectre
	Pxx, freqs, bins, im = specgram(sound_data, NFFT=NFFT, noverlap=128 , Fs=fs)
	show()

	#Siflantes detector
	nb_of_windows = Pxx.shape[1]
	nb_of_fft_coefs = Pxx.shape[0]

	#Compute the gaussian vector and plot
	weights = weighting_vector(nb_of_fft_coefs, sigma, mu, fs)
	f_wweights = np.linspace(0, fs/2, len(weights), endpoint=True)
	plot(f_wweights, weights)
	show()

	fft_coeficients = np.zeros(nb_of_fft_coefs)
	sibilant_desc = []
	weighted_ffts = [] 

	#Multiply the weights and the spectrum and show the multiplication
	for i in range(nb_of_windows):
	    weighted_fft = Pxx[:, i] * weights

	    if len(weighted_ffts) == 0:
	        weighted_ffts = weighted_fft
	    else:
	        weighted_ffts = np.c_[weighted_ffts, weighted_fft]
	        
	    sibilant_desc.append(sum(weighted_fft))
	    
	imshow(weighted_ffts, interpolation='nearest', aspect='auto')
	show()

	#Now mean the matrix to have only one descriptor
	sibilant_desc = [float(i)/max(sibilant_desc) for i in sibilant_desc]
	plot(sibilant_desc)
	show()

	#export audio
	max_index, max_value = max(enumerate(sibilant_desc), key=operator.itemgetter(1))
	wavwrite(sound_data[(max_index-5)*NFFT:(max_index+5)*NFFT], 'test.wav', fs=44100)

