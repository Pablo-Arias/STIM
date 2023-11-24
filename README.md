# STIM

STIM is a set of scripts and modules to handle the creation and transformation of audio and video experimental stimuli.

They allow to easilly handle indexing of audio/video databases, transform audio and video in different ways and extract different audio features.

There a series of tutorials in the tutorial section showing how these modules work.

Some interesting modules:

## audio_analysis ##
Extract audio features

## super_vp_commands ##
A set of wraper functions around IRCAM super-vp library 

## video_processing ##
A set of functions to handle video indexing, audio replacement in video, resolution changes, and more...

## Instalation and dependencies ##
These python modules depend on several externals for them to work properly. Moreover, if you want the full functionality of STIM you will also need to install IRCAM super-vp command line library (get it here : https://forum.ircam.fr/projects/detail/analysissynthesis-command-line-tools/).

Install other external dependencies with [homebrew](https://brew.sh/):
```
brew install ffmpeg opencv libsndfile cmake autoconf automake swig
```

To install STIM, I recomend using conda package manager. In the following, replace STIM_FOLDER with the path to the STIM folder.
```
cd STIM_FOLDER
git clone https://github.com/Pablo-Arias/STIM.git
```

Create a new conda environment with all dependencies, activate and add the path to your python path:
```
conda create --name stim39 python=3.9
conda activate stim39
conda develop STIM_FOLDER

# Install dependencies:
conda install -c roebel easdif
pip install scipy soundfile pyloudnorm pandas pyo praat-parselmouth matplotlib numpy opencv-python
```

Run unit tests to see if everything worked:
```
cd PATH_TO_STIM_FOLDER/tests/
python -m unittest discover .
```

Check that the tests are working and that packages are not missing.

## Tutorials ##
Now you can check the example modules to learn how to use these scripts.

Just open the .ipybn files inside the folder notebook_examples of the STIM (even from a browser).
