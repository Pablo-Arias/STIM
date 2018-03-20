# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ---------- Made by Pablo Arias @ircam on 03/2017
# ---------- transform a sound with the smile algorithm
# ----------
# ---------- import sys
# ---------- sys.path.append('/Users/arias/Documents/Developement/Python/')
# ---------- from one_way_smile_algorithm.one_way_smile_algorithm import transform_to_smile
# ----------
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#

#imports
from __future__ import division
import subprocess
from scikits.audiolab import wavread, wavwrite, Sndfile
import pandas as pd
import numpy as np
import os
import sys
from glob import glob
from os.path import basename
from pyo import *
from bisect import bisect


#plots
from pylab import *
import matplotlib.pyplot as plt
plt.style.use('ggplot')


#local packages
from plot_features import plot_mean_true_env_from_audio
from transform_audio import wav_to_mono, aif_to_wav
from parse_sdif import get_f0_info, get_formants_info
from super_vp_commands import freq_warp
from audio_analysis import mean_formant_from_audio, get_formant_data_frame

#---- get_formant_frequency_automation
def get_formant_frequency_automation(time_list, formant_freqs, min_val, max_val):
    #Fiter's Frequency automation
    if len(time_list) == len(formant_freqs):
        pairs = []
        nb_of_samps_per_mean = 8
        mean_time  = 0
        mean_freqs = 0 

        for i in range(len(formant_freqs)):
            mean_time = mean_time + time_list[i]
            mean_freqs = mean_freqs + formant_freqs[i]
            
            if i%nb_of_samps_per_mean == 0:
                time = mean_time / nb_of_samps_per_mean
                new_freq = mean_freqs / nb_of_samps_per_mean

                if min_val < new_freq and new_freq < max_val:
                    pairs.append((time, new_freq))
                
                mean_time = 0
                mean_freqs = 0
    
        return pairs
    else:
        print 'Error : freqs and time index don\'t have the length '

#---- get_gain_automation
def get_gain_automation(time_list, harmonicity_list, max_gain, neutral_gain, boost_when_harmonic):
    #Fiter's Frequency automation
    if len(time_list) == len(harmonicity_list):
        
        pairs = []      
        for i in range(len(time_list)):
            time = time_list[i]
            # Take only one every 12 in order to have more smooth transitions.
            if i%8 == 0:
                if harmonicity_list[i] == boost_when_harmonic:
                    pairs.append((time, max_gain[i]))
                else:
                    pairs.append((time, neutral_gain))
        
        return pairs

    else:
        print 'Error : harmonicity_list and time index don\'t have the length '        

# -- dynamic_filter
# -- boost/cut (depending on alpha) the third formant when the signal is harmonic and boost/cut when the signal is inharmonic at 6000 Hz
def dynamic_filter(s, audio_file, output_file, features_df, alpha):
    # ---------- Parameter init
    #harmonic filter Q
    Q           = 1.5
    dynamic_alpha_transf = isinstance(alpha, list)

    #for non harmoncic content the eq is:
    non_harmonic_eq_Q           = 1.8
    non_harmonic_eq_fr          = 6000


    # # ---------- global parameters ----------
    time_index = 0
    
    # # ---------- file parameters ----------
    time_alpha_array  = alpha[0]
    alpha_array       = alpha[1]
    gains             = [20*(alpha-1) for alpha in alpha_array]
    duration                = sndinfo(audio_file)[1]
    
    # # --------- sound parameters ----------
    formant3                = features_df["F3"] # Only formant 3 for now ...
    formant_3_freqs         = list(formant3.values)
    time_list               = list(formant3.index)

    #Resample gains to good time set
    gains                = [get_correspondent_time_taged_value(time, time_alpha_array, gains) for time in time_list]
    non_harmonic_eq_gain    = gains

    harmonicity_list        = features_df["is_harmonic"].values

    min_f3_freq_val            = 1000 #This is the minimum frequency for automating the filter
    max_f3_freq_val            = 5000 #This is the maximum frequency for automating the filter
    formant_3_freq_automation  = get_formant_frequency_automation(time_list, formant_3_freqs, min_val = min_f3_freq_val , max_val = max_f3_freq_val)
    formant_3_gain_automation  = get_gain_automation(time_list, harmonicity_list, gains, neutral_gain = 0, boost_when_harmonic = True)
    inharmonic_gain_automation = get_gain_automation(time_list, harmonicity_list, non_harmonic_eq_gain, neutral_gain = 0, boost_when_harmonic = False)


    # # --------- audio processing ----------
    s = s.boot()
    s.recordOptions(dur = duration
                    , filename = output_file
                    , fileformat = 0
                    , sampletype = 0
                    )

    sf    = SfPlayer(audio_file, speed = 1, loop = False)
    
    # Automations
    formant_3_freq_automation   = Linseg(formant_3_freq_automation, loop=False)
    formant_3_gain_automation   = Linseg(formant_3_gain_automation, loop=False)
    inharmonic_gain_automation  = Linseg(inharmonic_gain_automation, loop=False)
    formant_3_freq_automation.play()
    formant_3_gain_automation.play()
    inharmonic_gain_automation.play()

    # singal chain
    eq              = EQ(sf, freq = formant_3_freq_automation, q = Q, boost = formant_3_gain_automation, type = 0, mul = 0.8)
    non_harmonic_eq = EQ(eq, freq = non_harmonic_eq_fr, q = non_harmonic_eq_Q, boost = inharmonic_gain_automation, type = 0, mul = 0.8)
    out             = non_harmonic_eq.mix(2).out()

    #aux   = CallAfter(callback, time_list)
    s.start()
    #s.stop()
    #s.shutdown()

def get_correspondent_time_taged_value(time_tag, time_alpha_array, alpha_array):
    index = bisect(time_alpha_array, time_tag)
    return alpha_array[index-1]

def generate_warping_file(features_df, audio_file, alpha):
    time_alpha_array = alpha[0]
    alpha_array      = alpha[1]

    #This function generates a warping file when alpha is dynmaic
    x, fs, enc = wavread(audio_file)
    nbpf = 7 # 5 formants + 2 neutral points + 2 begining and end spectrum

    #name the warp file
    bpf_tag = '_warpfile_for_freq_warp.txt'
    if os.path.dirname(audio_file) == "":
        file_tag = os.path.basename(audio_file)
        file_tag = os.path.splitext(file_tag)[0]
        warp_name = file_tag + bpf_tag
    else:
        file_tag = os.path.basename(audio_file)
        file_tag = os.path.splitext(file_tag)[0]
        warp_name = os.path.dirname(audio_file)+ "/" + file_tag + bpf_tag


    #mean of each formant
    features_df = features_df.loc[features_df["is_harmonic"] == True]

    F1_mean = features_df.mean()["F1"]
    F2_mean = features_df.mean()["F2"]
    F3_mean = features_df.mean()["F3"]
    F4_mean = features_df.mean()["F4"]
    F5_mean = features_df.mean()["F5"]

    # Create bpf file
    with open( warp_name, 'w') as f:
        #write number of bpf
        f.write(str(nbpf)+'\n')
        for time, row in features_df.iterrows():
            if row["is_harmonic"]:
                alpha = get_correspondent_time_taged_value(time_tag = time, time_alpha_array = time_alpha_array, alpha_array = alpha_array)

                f.write(str(time) #Time
                        + ' '          
                        + str(0) + ' ' + str(0) #0 to 0
                        + ' '          
                        + str(F1_mean/2) + ' ' + str(F1_mean/2) # 1st Neutral point
                        + ' '          
                        + str(F2_mean) + ' ' + str(F2_mean * alpha)#str((freqs_F1[i] * F1_coef) #1st formant bp
                        + ' '          
                        + str(F3_mean) + ' '+ str(F3_mean  * alpha)#3rd formant bp
                        + ' '
                        + str(F5_mean) + ' ' + str(F5_mean) #5rd formant bp
                        + ' '
                        + str(10000) + ' ' + str(10000) #neutral bp at end of voice 
                        + ' '
                        + str(fs/2) + ' ' + str(fs/2) #Last bpf
                        + '\n'
                       )
                #very_adaptive_freq_warp
                #f.write(str(time) #Time
                #        + ' '          
                #        + str(0) + ' ' + str(0) #0 to 0
                #        + ' '          
                #        + str(row["F1"]/2) + ' ' + str(row["F1"]/2) # 1st Neutral point
                #        + ' '          
                #        + str(row["F2"]) + ' ' + str(row["F2"] * alpha)#str((freqs_F1[i] * F1_coef) #1st formant bp
                #        + ' '          
                #        + str(row["F3"]) + ' '+ str(row["F3"]  * alpha)#4rd formant bp
                #        + ' '
                #        + str(row["F5"]) + ' ' + str(row["F5"]) #5rd formant bp
                #        + ' '
                #        + str(10000) + ' ' + str(10000) #neutral bp at end of voice 
                #        + ' '
                #        + str(fs/2) + ' ' + str(fs/2) #Last bpf
                #        + '\n'
                #       )

    return warp_name



def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx


# ---------- plot to a file the sound features used in the algorithm
def plot_file_info(features_df, sound_filename, harmonicity_threshold, alpha):
    features_df  = features_df.reset_index()
    time         = features_df["time"].values
    F1           = features_df["F1"].values
    F2           = features_df["F2"].values
    F3           = features_df["F3"].values
    F4           = features_df["F4"].values
    F5           = features_df["F5"].values
    f0           = features_df["f0_val"].values
    f0harm       = features_df["harm"].values

    #waveform
    file_format = 'pdf'
    sound_data, fs, enc = wavread(sound_filename)
    t = np.linspace(0, len(sound_data)/fs, len(sound_data), endpoint=True)

    fig = plt.figure()

    #Waveform
    plt.subplot(511)
    ylabel('Amplitude')
    title('Waveform')
    grid(True)
    plt.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom='off',      # ticks along the bottom edge are off
                    top='off',         # ticks along the top edge are off
                    labelbottom='off'  # labels along the bottom edge are off
                    )

    plt.plot(t, sound_data)

    #F0
    plt.subplot(512)
    ylabel('Frequency')
    grid(True)
    plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off'  # labels along the bottom edge are off
                )    
    plt.semilogy(time, f0, color = 'green', label = 'f0')
    legend(loc='best', prop={'size':6})


    #1st Formant
    plt.subplot(513)
    ylabel('Frequency')
    grid(True)
    plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off'  # labels along the bottom edge are off
                )
    plt.semilogy(time, F1, label = 'F1')


    #2nd Formant
    plt.semilogy(time, F2, label = 'F2')
    
    #3rd Formant
    plt.semilogy(time, F3, label = 'F3')

    #4th Formant
    plt.semilogy(time, F4, label = 'F4')

    #5th Formant
    plt.semilogy(time, F5, label = 'F5')
    legend(loc='best', prop={'size':6}) 



    #Harmonicity
    plt.subplot(514)
    title('Harmonicity')
    ylabel('%')
    xlabel('time (s)')
    grid(True)
    plt.plot(time, f0harm, label = 'Harmonicity')

    #Treshold
    harmonic_vector = harmonicity_threshold * ones(len(time))
    plt.plot(time, harmonic_vector, label = 'Harm Treshold')

    #Alpha
    plt.subplot(515)
    title('Alpha')
    ylabel('a.u.')
    xlabel('time (s)')
    grid(True)
    time_alpha_array = alpha[0]
    alpha_array      =  alpha[1]
    alpha = [get_correspondent_time_taged_value(t, time_alpha_array, alpha_array) for t in time]
    
    plt.plot(time, alpha, label = 'Harmonicity')


    legend(loc='best', prop={'size':6})

# --------------------------------------------------
# --------------------------------------------------
# ---------- transform_to_smile --------------------
# ------ Main transformation algorithm -------------
# --------------------------------------------------
# --------------------------------------------------
def transform_to_smile(audio_file
                            , target_file
                            , alpha                    = 1.25
                            , feature_window           = 10
                            , harmonicity_threshold    = 0.8
                            , plot_features_pdf        = False
                            , delete_warp_file         = True
                            , formant_estimation       = "lpc"
                            , do_dynamic_filter        = True
                            , freq_warp_ws             = 512
                            , lpc_order                = 25
                            , warp_method              = "lpc"
                            , freq_warp_oversampling   = 64
                        ):
    """
    Transforms a mono file using the smile algorithm
    parameters:
        audio_file             : audio_file to transform
        target_file            : target audio file to create
        alpha                  : alpha parameter which controls both the amount of shift and of gain boost, alpha = 1.25 is equivalent to the smile algorithm, alpha = 0.8 is the pursed transofrmation
                                 For dynamic transformations, alpha can now also be a two dimension list with time (in seconds) and alpha values

        feature_window         : how many samples to mean formant frequencies over time
        harmonicity_threshold  : what harmonicity threshold to take to consider when the signal is harmonic
        plot_features_pdf      : plot the features used by ziggy for the overal tranformation. This is usefull to see if everything is estimated correctly when transforming a sound
        formant_estimation     : can be either lpc, t_env or praat, the method used for the IEEE paper and the Curr Bio is LPC
        do_dynamic_filter      : If set to True, apply the dynamic filter t the 3rd formant, as well as the inharmonic filter
                                if False, no filteringis performed
        freq_warp_ws           : Frequency warping windows size
        lpc_order              : lpc order for the warping stage
        warp_method            : this parameter is either "t_env" or "lpc", 
                                  and specifies what enevelope estimation to use in the warping stage
                                  the default is lpc
        freq_warp_oversampling : Oversampling of the frequency warping

    output : 
        A file called target file transformed with the smile effect

    warning : 
        this function only works for mono files    
    """
    #if alpha is an int or a float, change it to an array with time and alpha
    if isinstance(alpha, float) or isinstance(alpha, float):
        alpha = [[1], [alpha]]

    warped_file   = "warped_file.wav"
    filtered_file = target_file

    #pyo server init
    s  = Server(duplex=0, audio="offline")

    #generate formant analysis
    if formant_estimation == "lpc":
        features_df = get_formant_data_frame(audio_file, nb_formants = 5 ,t_env_or_lpc = "lpc", destroy_sdif_after_analysis = True)
    elif formant_estimation == "t_env":
        #TODO : TEST the true env extraction of formants
        features_df = get_formant_data_frame(audio_file, nb_formants = 5 ,t_env_or_lpc = "t_env", destroy_sdif_after_analysis = True)

    elif formant_estimation == "praat":
        from audio_analysis import get_formant_ts_praat
        freqs_df, bws_df = get_formant_ts_praat(audio_file)
        features_df = freqs_df.reset_index().set_index("time")
    else:
        print "formant_estimation should be either lpc, t_env or praat"

    features_df = features_df[["frequency", "Formant"]].reset_index()

    features_df = features_df.pivot(index='time', columns='Formant', values='frequency')

    #generate f0 analysis
    from audio_analysis import get_f0
    f0times, f0harms, f0vals = get_f0(audio_file)
    
    #merge harmonicity and formant values
    new_harm_vals = []
    new_f0_vals = []

    tests         = []
    new_harm_vals = []
    new_f0_vals   = []
    for time in features_df.index:
        index = find_nearest(array = f0times, value = time)
        new_harm_vals.append(f0harms[index])
        new_f0_vals.append(f0vals[index])
        
    features_df['harm'] = new_harm_vals
    features_df['f0_val'] = new_f0_vals

    #window features
    features_df = features_df.rolling(window=feature_window, center=True).mean().dropna()
    features_df["is_harmonic"] = np.where(features_df['harm'] > harmonicity_threshold, True, False)
    
    #generate warping bpf file
    warp_file = generate_warping_file(features_df, audio_file, alpha)

	#frequency warping
    freq_warp(audio_file, warped_file, warp_file, freq_warp_ws = freq_warp_ws,lpc_order=lpc_order, warp_method=warp_method ,wait = True, freq_warp_oversampling= freq_warp_oversampling)

	#dynamic filter
    if do_dynamic_filter:
        dynamic_filter(s = s, audio_file = warped_file , output_file = filtered_file, features_df = features_df, alpha = alpha)
        s.shutdown()
    else:
        #copy file without filtering
        from shutil import copyfile
        print "No dynamic filter applied"
        copyfile(warped_file, filtered_file)

    if plot_features_pdf:
        plot_file_info(features_df = features_df, sound_filename = audio_file,harmonicity_threshold = harmonicity_threshold, alpha=alpha)
        plt.savefig( filtered_file+".pdf" , dpi = 350)

    if delete_warp_file:
        os.remove(warp_file)
        os.remove(warped_file)


