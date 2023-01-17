
from super_vp_commands import transient_detection_analysis

def onset_detction(file, rms_threshold):
	onsets = []
	return onsets


def happy_transform(file):
	return
	#onset detection


	#Inflection profile
	
	#At each onset transpose sound with pattern

def transform_file():
	return
	#onset detection


if __name__ == "__main__":
	audio_file = "/Users/arias/Desktop/Projects/Alexander_tagesson/database/audio/denoised/ID112_vid1_crop_downsized.wav"

	#transient_detection_analysis(audio_file)

	import librosa
	x, sr = librosa.load(audio_file)
	onset_frames = librosa.onset.onset_detect(x, sr=sr, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1)

	print(onset_frames)


