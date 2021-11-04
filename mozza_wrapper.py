from __future__ import absolute_import
from __future__ import print_function
from subprocess import call
import shlex, subprocess
import os
import numpy as np
from shutil import copyfile
from conversions import get_file_without_path

mozza_path = "/Users/arias/Documents/Developement/ducksoup/plugin_dev/code_repo/mozza"



def create_deformation_file(container_folder, source, model, output, wait=True):
	cmd 		= "docker run -v "+container_folder+":/data -v "+mozza_path+":/mozza-path creamlab/mozza mozza-templater -b -m /mozza-path/data/in/shape_predictor_68_face_landmarks.dat -n /data/"+source+" -s /data/"+model+" -o /data/"+output

	args 		= shlex.split(cmd)

	p = subprocess.Popen(args)
	if wait:
		p.wait() #wait	

def transform_img_with_mozza(container_folder,  source, target, wait=True, deformation_file="./default.dfm", alpha=1.0, face_thresh=0.25 , overlay=False , beta=0.1, fc=5.0):	
	cmd 		= "docker run -v "+container_folder+":/data creamlab/mozza:latest gst-launch-1.0 filesrc location=/data/"+source+" ! decodebin ! videoconvert ! mozza deform=/data/"+deformation_file+" alpha="+str(alpha)+" ! videoconvert ! jpegenc ! filesink location=/data/"+target
	args 		= shlex.split(cmd)

	p = subprocess.Popen(args)
	if wait:
		p.wait() #wait	

def transform_video_with_mozza(container_folder,  source, target, wait=True, deformation_file="./default.dfm", alpha=1.0, face_thresh=0.25 , overlay=False , beta=0.1, fc=5.0):	
	#cmd 		= "docker run -v "+container_folder+":/data creamlab/mozza:latest gst-launch-1.0 filesrc location=/data/"+source+" ! decodebin ! videoconvert ! mozza deform=/data/"+deformation_file+" alpha="+str(alpha)+" ! videoconvert ! x264enc ! mp4mux name=mux ! filesink location=/data/"+target
	
	#cmd 		= "docker run -v "+container_folder+":/data creamlab/mozza:latest gst-launch-1.0 filesrc location=/data/1F_1.mp4 ! decodebin ! videoconvert ! x264enc ! mp4mux ! filesink location=/data/output.mp4"

	cmd			= "docker run --env GST_DEBUG=2 -v "+container_folder+":/data creamlab/mozza:latest gst-launch-1.0 filesrc location=/data/"+source+" ! decodebin ! videoconvert ! mozza deform=/data/"+deformation_file+" alpha="+str(alpha)+" ! videoconvert ! x264enc ! mp4mux ! filesink location=/data/"+target


	print(cmd)
	
	args 		= shlex.split(cmd)

	p = subprocess.Popen(args)
	if wait:
		p.wait() #wait				

		#gst-launch-1.0 filesrc location=input.mkv ! decodebin ! videoconvert ! x264enc ! mp4mux name=mux ! filesink location=output.mp4


#Create deformation
# container_folder = "/Users/arias/Desktop/container"
# source = "neutral.png"
# model = "smile.png"
# deformation_file = "test.dfm"
# create_deformation_file(container_folder = container_folder, source=source, model=model, output=deformation_file)

#Transform img with mozza
#container_folder = "/Users/arias/Desktop/container"
#source = "neutral.png"
#target = "transformed.jpg"
#deformation_file = "test.dfm"
#transform_img_with_mozza(container_folder = container_folder,  source = source, target= target, deformation_file=deformation_file, alpha=1, face_thresh=0.25 , overlay=False , beta=0.1, fc=5.0)

# #Create deformation
container_folder = "/Users/arias/Desktop/container"
source = "1F_1.mp4"
source = "1_25_V-A-.avi"
target = "output.avi"
deformation_file = "test.dfm"


transform_video_with_mozza(container_folder = container_folder,  source = source, target= target, deformation_file=deformation_file, alpha=-1.0, face_thresh=0.25 , overlay=False , beta=0.1, fc=5.0)


