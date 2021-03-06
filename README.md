# STIM

This series of scripts and python modules were done to handle the creation and transformation of audio and video experimental stimuli.

They allow to easilly handle indexing of audio/video databases, transform audio and video in different ways and extract different audio features.

There a series of tutorials in the tutorial section showing how these modules work.

Some interesting modules:

## audio_analysis ##
Extract audio features

## super_vp_commands ##
A set of wraper functions around IRCAM super-vp library (needs audiosculpt http://forumnet.ircam.fr/product/audiosculpt-en/)

## video_processing ##
A set of functions to handle video indexing, audio replacement in video, resolution changes, and more...

## Instalation and dependencies ##
These python modules depend on several externals for them to work properly. If you want to use the super_vp_commands or parse sdif files you should isntall all of the above packages.


### Download ###
Download the STIM folder (or clone it with git) and put it somewhere safe in your computer — where it will live forever witout being moved. 
Now you need your python install to see the STIM folder. To do this you can do one of these :
* If you are using anaconda, create a file called "STIM_python_module.pth" with the path to the STIM folder inside, that's all. If you are using a mac save it in "/Users/your_user_name/anaconda/lib/python2.7/site-packages" if you are using windows, save it in "path_to_anaconda/anaconda/lib/python2.7/site-packages".

* If you are not using conda, you can import these python modules each time, by typing the following commands at the begining of your script, and changing the PATH_TO_STIM_FOLDER by your actual path to the STIM folder: 
```
import sys
sys.path.append('PATH_TO_STIM_FOLDER')
from audio_analysis import 'the functions you need' #example of import
```

### Install Homebrew ### 
Check the Info [here](https://brew.sh/) to install homebrew.

###  Install libsnd ### 
```
brew install libsndfile
brew install cmake
brew install autoconf automake
brew install swig
```
###  Install sckits.audiolab toolbox ### 
If you are using conda, you can try :
```
conda install -c weiyan scikit-audiolab
```

If you are not using conda, try installing it with pip:
```
pip install scikits.audiolab
```

###  Install pyo ### 
Pyo is a set of python modules used for RT and offline audioprocessing. To install it download the installer [here](http://ajaxsoundstudio.com/software/pyo/)

###  Install Audiosculpt ### 
These scripts use Audiosculpt to generate and parse audio analysis. 
For them to work properly, make sure you installed Audiosculpt from the [ircam forum](http://forumnet.ircam.fr/)

Once you installed Audiosculpt, you should update the "Audiosculpt version" in the script super_vp_path inside of the STIM folder so that the python module can call Audiosculpt.


###  Install EASDIF ### 
Now on to the fun part,
To parse the files generated by audiosculpt automatically with python, (.sdif files) you will need the EASDIF library, which can be installed with the following commands:

```
cvs -d:pserver:anonymous@sdif.cvs.sourceforge.net:/cvsroot/sdif login 
cvs -z3 -d:pserver:anonymous@sdif.cvs.sourceforge.net:/cvsroot/sdif co -P EASDIF_SDIF
cd EASDIF_SDIF
mkdir -p build_EASDIF_x86_64
cd build_EASDIF_x86_64
rm -f CMakeCache.txt
cmake -DEASDIF_DO_PYTHON:bool=on -DPYTHON:STRING=python -DEASDIF_BUILD_STATIC:BOOL=ON ..
make Easdif_static_pic
make install_python_easdif_module_globally
```
or if you have no right to write into your python installation
```
make install_python_easdif_module_locally
```

## Tutorials ##
Now you can check the example modules to learn how to use these scripts.

Just open the .ipybn files inside the folder notebook_examples of the STIM (even from a browser).
