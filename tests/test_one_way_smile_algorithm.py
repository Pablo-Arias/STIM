from __future__ import absolute_import
from __future__ import print_function
import unittest
from one_way_smile_algorithm.one_way_smile_algorithm import *
from transform_audio import wav_to_mono
import soundfile
import os

## -------------------- ##
##  Test Smile algorithm
## -------------------- ##
class Test_one_way_smile_algorithm(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.audio_folder = "Sounds/"
		cls.audio_source = cls.audio_folder + 'F11h1.wav'
		cls.aux_file     = cls.audio_folder + "aux_1234456.wav"

	def test_transform_to_smile(self):
		wav_to_mono(self.audio_source, self.aux_file)
		transform_to_smile(audio_file = self.aux_file, target_file= self.aux_file)
		os.remove(self.aux_file)

if __name__ == '__main__':
	unittest.main()