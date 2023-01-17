from __future__ import absolute_import
from __future__ import print_function
import unittest
from transform_audio import *
import soundfile
import os
import glob

## -------------------- ##
##  Tranform_audio tests
## -------------------- ##
class Test_transform_audio(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.audio_folder = "Sounds/"
		cls.audio_source = cls.audio_folder + 'F11h1.wav'
		cls.aux_file     = cls.audio_folder + "aux_1234456.wav"

	def test_wav_to_mono(self):
		wav_to_mono(self.audio_source, self.aux_file)
		f = soundfile.SoundFile(self.aux_file)
		
		#Test that file exists
		self.assertTrue(os.path.exists(self.aux_file))

		#Test that number of channels is correct
		self.assertEqual(f.channels, 1)
		os.remove(self.aux_file)
	
	def test_mono_to_stereo(self):
		wav_to_mono(self.audio_source, self.aux_file)
		mono_to_stereo(self.aux_file, self.aux_file)
		f = soundfile.SoundFile(self.aux_file)

		#Test that file exists
		self.assertTrue(os.path.exists(self.aux_file))		

		#Test that files number of channels is correct
		self.assertEqual(f.channels, 2)
		os.remove(self.aux_file)

	def test_format_to_format(self):
		aux = os.path.splitext(self.audio_source)[0]+".aiff"
		format_to_format(self.audio_source, aux)
		
		#Test that file exists
		self.assertTrue(os.path.exists(aux))

		#Test that format is aiff
		f = soundfile.SoundFile(aux)
		self.assertEqual(f.format, 'AIFF')

		#remove file
		os.remove(aux)

	def test_extract_sentences_tags(self):
		tags, lengths = extract_sentences_tags(self.audio_folder+"as.wav")
		
		# Test it extracted some tags
		self.assertGreater(len(tags), 0)
		
		# Test if lengths of extracted sentences are ok
		for length in lengths:
			self.assertIsNotNone(length)	

	def test_index_wav_file(self):
		index_folder = self.audio_folder + "Indexed/"
		#remove from folder
		for file in glob.glob(index_folder+"*.wav"):
			os.remove(file)
		index_wav_file(self.audio_folder + "as.wav", rms_threshold=-40, target_folder=index_folder)
		nb_files = len(glob.glob(index_folder+"*.wav"))
		self.assertNotEqual(0, nb_files)
		
		#remove all files
		for file in glob.glob(index_folder+"*.wav"):
			os.remove(file)

	def test_filter_file(self):
		#filter_file Example 
		from pyo import Server
		s  = Server(duplex=0, audio="offline")
		filter_file(s, self.audio_source, self.aux_file, type=0, freq=300, q =1)
		self.assertTrue(os.path.exists(self.aux_file))

		#remove file
		os.remove(self.aux_file)

	def test_cut_silence_in_sound(self):
		aux_source = "aux_mono.wav"
		wav_to_mono(self.audio_source, aux_source)
		cut_silence_in_sound(source = aux_source, target = self.aux_file, rmsTreshhold = -55, WndSize = 128)
		aux, fs    = soundfile.read(self.aux_file)  
		source, fs = soundfile.read(aux_source)
		self.assertLess(len(aux), len(source))

		#Remove files used
		os.remove(self.aux_file)
		os.remove(aux_source)

	def test_audio_peak_normalisation(self):
		audio_peak_normalisation(self.audio_source, self.aux_file)
		self.assertTrue(os.path.exists(self.aux_file))
		os.remove(self.aux_file)


if __name__ == '__main__':
	unittest.main()
