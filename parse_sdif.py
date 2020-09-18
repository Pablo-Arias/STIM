# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
# ---------- Made by pablo Arias @ircam on 11/2015
# ---------- Analyse audio and return soudn features
# ---------- to us this don't forget to include these lines before your script:
# ----------
# ----------
# --------------------------------------------------------------------#
# --------------------------------------------------------------------#
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from pandas import DataFrame
import os

#sdif
import eaSDIF
import sys
from fileio.sdif.FSdifLoadFile import FSdifLoadFile
from six.moves import range


def get_f0_info(f0file) :
    """
    load f0 from ascii or SDIF file 
    return tuple of np.arrays as follows
    return f0times, f0harm, f0val
    f0harm will be None if input is file is ASCII format
    """
    try:
        (f0times, f0data) = FSdifLoadFile(f0file)
        f0times = np.array(f0times)
        dd = np.array(f0data)
        if len(dd.shape) > 1 and dd.shape[1] > 2 :
            f0harm  = dd[:,2]
        else:
            f0harm = None
        f0val   = np.array(f0data)[:,0]
    except RuntimeError as rr :
        print("failed reading "+f0file+" as sdif try reading as txt!")
        f0times_data = np.loadtxt(f0file)
        f0times = np.array(f0times_data)[:,0]
        f0val   = np.array(f0times_data)[:,1]        
        f0harm  = None        
        
    return f0times, f0harm, f0val

def get_nb_formants(formant_file):
    """
    get the number of formants of an sdif file
    """
    try:      
        ent = eaSDIF.Entity()
        ent.OpenRead(formant_file)
        frame = eaSDIF.Frame()
        #take the length of the fifth matrix, sometimes the first ones don't have the good number of formants =P
        ent.ReadNextSelectedFrame(frame)
        ent.ReadNextSelectedFrame(frame)
        ent.ReadNextSelectedFrame(frame)
        ent.ReadNextSelectedFrame(frame)
        ent.ReadNextSelectedFrame(frame)
        try :
            mat = frame.GetMatrixWithSig("1RES")
        except IndexError :
            pass
        
        par_mat = mat.GetMatrixData()

        return len(par_mat)
    except EOFError : 
        pass
    
    return 0


def get_formants_info(formant_file):
    """
    load formant_file from SDIF file 
    return an array of panadas data frames with the formants in the sdif file
    
    return: Array of pandas data frames with formants
    """

    ts = [] # analysis times
    cols_names = ("Frequency", "Amplitude", "Bw", "Saliance")
    nb_formants = get_nb_formants(formant_file)

    try:
        formants = []
        for i in range(nb_formants):
            formant = DataFrame(columns=cols_names)
            formants.append(formant)


        ent = eaSDIF.Entity()
        ent.OpenRead(formant_file)
        frame = eaSDIF.Frame()
        ent.ReadNextSelectedFrame(frame)
    except EOFError : 
        print("In get_formants_info first exception")
        pass 


    try:

        while  ent.ReadNextSelectedFrame(frame):
            try :
                mat = frame.GetMatrixWithSig("1RES")
            except IndexError :
                print("Index Error dans get_formants_info in parse_sdif")
                # matrix is not present so we continue
                continue
            frame_time = frame.GetTime()            
            ts.append(frame_time)
            par_mat = mat.GetMatrixData()            
            if par_mat.shape[1] != 4 :
                raise RuntimeError("partial data number of columns "+str(cols)+" unequal to 4 !")
            
            if len(par_mat) == nb_formants:
                for i in range(nb_formants):
                    formants[i] = formants[i].append(DataFrame([par_mat[i].tolist()], columns=cols_names, index = [frame_time]))

        
    except EOFError : 
        pass
    return formants

def get_matrix_values(sdif):
    """
    load data from ascii or SDIF file 
    return time-tagged values and matrix data
    return tlist, Matrix_data
    This can be used to extract data from lpc or true env .sdif files
    """
    inent = eaSDIF.Entity()
    res = inent.OpenRead(sdif);
    if res == False:
        raise RuntimeError("get_lpc:: "+ sdif +" is no sdif file or does not exist")
    dlist = [];
    tlist = [];
    vec = eaSDIF.Vector()
    frame = eaSDIF.Frame()

    #fft size
    #intypes = inent.GetTypeString() # Very practical line fr printing what is inside    
    for frame in inent:
        has_IGBG = frame.MatrixExists("IGBG")
        if has_IGBG:
            mat = frame.GetMatrixWithSig("IGBG")
            mat.GetRow(vec, 0)
            sample_rate = vec[1]
            fftsize = vec[3]
    fftsize = int(fftsize / 2)
    
    #Extract time tag values
    for frame in inent:
        mat  = frame.GetMatrix(1);
        nrow = mat.GetNbRows();
        ncol = mat.GetNbCols();        
        if nrow > 1  and ncol > 0 :
            tlist.append(frame.GetTime());

    #Extract Matrix data values
    for frame in inent:
        for i in range (0,fftsize + 1):
            mat  = frame.GetMatrix(1);
            nrow = mat.GetNbRows();
            ncol = mat.GetNbCols();
            if nrow > 1  and ncol >= 0 :
                mat.GetRow(vec, i)
                dlist.append(float(np.array((vec)[0])));

    #Convert dlist into a matrix
    sample_nb = len(tlist) - 1 # Because the first value in tlist should be ignored
    fftsize_range = fftsize + 1
    sample_nb_range =sample_nb + 1
    matrix_data = np.zeros((sample_nb_range, fftsize_range))
    for row in range (0, sample_nb_range):
        for col in range (0, fftsize_range):
            matrix_data[row][col] = dlist[row*(fftsize_range) + col]

    #when using the flag -OS1 in super vp the amplitude values are in linear, here we transform it to db so the amplitudes are in db
    from conversions import lin2db
    matrix_data = lin2db(matrix_data)

    return tlist, matrix_data

def get_formants_parameter(sdif, parameter):
    """
    This is a function for quick access to the parameters of formants
    parameters can be:
        "Frequency", "Amplitude", "Bw", "Saliance"

    """
    formants = get_formants_info(sdif)
    parameters = []
    for formant in formants:
        parameters.append(formant[parameter])
    return parameters


#--------------------
#--------------------
#---------- Compute mean for sound features
#---------- 
#--------------------
#--------------------

def mean_matrix(sdif):
    tlist, matrix_data = get_matrix_values(sdif);
    matrix_data = matrix_data.mean(axis=0);
    return matrix_data

def formant_from_sdif(sdif):

    formants = get_formants_info(sdif)    
    return formants

def mean_formant_from_sdif(sdif):
    formants = get_formants_info(sdif)
    
    mean_formants =[]
    for formant in formants:
        mean_formants.append(formant.mean())

    return mean_formants

def median_formant_from_sdif(sdif):
    formants = get_formants_info(sdif)
    
    median_formants =[]
    for formant in formants:
        median_formants.append(formant.median())

    return median_formants    

def mean_pitch(sdif):
    f_name = analysis_path + subject_tag(sdif) + str(sdif) + f0_tag
    f0times, f0harm, f0val = get_f0_info(str(f_name)) 
    return f0val.mean(0)

