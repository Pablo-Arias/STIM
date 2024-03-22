# Install ffmpeg if not installed already : brew install ffmpeg
# cd to a PATH where you will keep my STIM repo
# Clone it there : git clone https://github.com/Pablo-Arias/STIM.git
# Clone it there : git clone https://github.com/Pablo-Arias/video_analysis.git
# Create an env to use it:
# conda create --name stim39 python=3.9
# conda activate stim39
# conda install -c roebel easdif
# conda develop PATH/STIM
# conda develop PATH/video_analysis
# pip install scipy soundfile pyloudnorm pandas pyo praat-parselmouth matplotlib numpy opencv-python

from ducksoup import ds_process_audio_only

import glob

for source_folder in glob.glob("data/audio_only/mkultimatum_game_template4/*/recordings/"):
    folder_tag = source_folder.split("/")[-3] + "/"
    ds_process_audio_only(source_folder=source_folder, folder_tag=folder_tag, preset="veryfast")