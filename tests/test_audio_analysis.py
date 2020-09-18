from __future__ import absolute_import
from __future__ import print_function
import unittest
from transform_audio import wav_to_mono
from audio_analysis import *
import soundfile
import os
import glob
import numpy as np

## -------------------- ##
##  Tranform_audio tests
## -------------------- ##
class Test_audio_analysis(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.audio_folder = "Sounds/"
		cls.audio_source = cls.audio_folder + 'as.wav'
		cls.aux_file     = cls.audio_folder + "aux_1234456.wav"

	def test_get_spectral_centroid(self):
		t, centroid_list = get_spectral_centroid(self.audio_source)
		
		#Test the extracted features
		self.assertTrue(len(t) >0  )
		self.assertTrue(len(centroid_list) >0 )
		self.assertTrue(np.mean(centroid_list) >0)

	def test_get_mean_spectral_centroid(self):
		centroid = get_mean_spectral_centroid(self.audio_source)
		self.assertTrue(centroid > 0 )

	def test_get_mean_spectral_centroid_when_sound(self):
		mean_sc = get_mean_spectral_centroid_when_sound(self.audio_source)

		#Check condition
		self.assertTrue(mean_sc > 0)

	def test_get_RMS_over_time(self):
		t, values = get_RMS_over_time(self.audio_source)
		
		#Test the extracted features
		self.assertTrue(len(t) > 0 )
		self.assertTrue(len(values) >0 )

	def test_get_intervals_of_sound(self):
		intervals = get_intervals_of_sound(self.audio_source)

		#Test
		self.assertTrue(len(intervals) > 0 )
		self.assertTrue(intervals[0][0] <  intervals[0][1])

	def test_Extract_ts_of_pitch_praat(self):
		times, f0s = Extract_ts_of_pitch_praat(self.audio_source)

		#Test
		self.assertTrue(len(times) > 0 )
		self.assertTrue(len(f0s) > 0 )

	def test_get_mean_pitch_praat(self):
		mean_pitch = get_mean_pitch_praat(self.audio_source)
		self.assertTrue(mean_pitch > 0 )

	def test_get_pitch_std(self):
		mean_pitch_std = get_pitch_std(self.audio_source)
		self.assertTrue(mean_pitch_std > 0 )

	def test_formant_from_audio(self):
		nb_formants = 5
		formant = formant_from_audio(self.audio_source, nb_formants = nb_formants)
		self.assertTrue(len(formant) == nb_formants )

	def test_get_tidy_formants(self):
		formants = get_tidy_formants(self.audio_source)
		self.assertIsNotNone(formants)

	def test_mean_formant_from_audio(self):
		formants = mean_formant_from_audio(audio_file = self.audio_source, nb_formants = 5)
		self.assertIsNotNone(formants)


	def test_get_formant_data_frame(self):
		formants = get_formant_data_frame(self.audio_source)
		self.assertIsNotNone(formants)

	def test_get_formant_ts_praat(self):
		freqs_df, bws_df = get_formant_ts_praat(self.audio_source)
		self.assertIsNotNone(freqs_df)
		self.assertIsNotNone(bws_df)

	def test_get_mean_formant_praat_harmonicity(self):
		freqs_df, bws_df = get_mean_formant_praat_harmonicity(self.audio_source)
		self.assertIsNotNone(freqs_df)
		self.assertIsNotNone(bws_df)

	def test_get_mean_formant_praat(self):
		formants = get_mean_formant_praat(self.audio_source)
		self.assertIsNotNone(formants)

	def test_get_mean_tidy_formant_praat(self):
		df = get_mean_tidy_formant_praat(self.audio_source)
		self.assertIsNotNone(df)

	def test_analyse_audio_folder(self):
		df_stimuli = analyse_audio_folder("test_folder_analysis/")
		self.assertIsNotNone(df_stimuli)

	def test_get_formant_dispersion(self):
		disp = get_formant_dispersion(self.audio_source)
		self.assertGreater(disp, 0)

	def test_estimate_vtl(self):
		vtl = estimate_vtl(self.audio_source)
		self.assertGreater(vtl, 0)

	def test_get_mean_lpc_from_audio(self):
		lpc = get_mean_lpc_from_audio(self.audio_source)
		self.assertIsNotNone(lpc)

	def test_get_true_env_analysis(self):
		t_env = get_true_env_analysis(self.audio_source)
		self.assertIsNotNone(t_env)

	def test_get_true_env_ts_analysis(self):
		t_env = get_true_env_ts_analysis(self.audio_source)
		self.assertIsNotNone(t_env)

	def test_get_spectrum(self):
		spectrum = get_spectrum(self.audio_source)
		self.assertIsNotNone(spectrum)

	def test_get_f0(self):
		f0times, f0harm, f0val = get_f0(self.audio_source)
		
		#Check if analysis extracted features
		self.assertGreater(len(f0times), 0)
		self.assertGreater(len(f0harm), 0)
		self.assertGreater(len(f0val), 0)

		#Check values inside
		self.assertGreater(np.mean(f0times), 0)
		self.assertGreater(np.mean(f0harm) , 0)
		self.assertGreater(np.mean(f0val)  , 0)		

	def test_get_zero_cross_from_data(self):
		zc = get_zero_cross_from_data(self.audio_source, 1024)
		self.assertIsNotNone(zc, 0)

	def test_get_rms_from_wav(self):
		rms = get_rms_from_wav(self.audio_source)
		self.assertIsNotNone(rms, 0)

	def test_get_sound_duration(self):
		sd = get_sound_duration(self.audio_source)
		self.assertGreater(sd, 0)

	def test_get_nb_channels(self):
		channels = get_nb_channels(self.audio_source)
		self.assertTrue(channels == 1)

	def test_find_silence(self):
		silence = find_silence(self.audio_source)
		self.assertIsNotNone(silence)

	def get_mean_rms_level_when_sound():
		rms = mean_rms_level_when_sound(self.audio_source)
		self.assertGreater(rms, 0)

if __name__ == '__main__':
	unittest.main()
