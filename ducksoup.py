from video_processing import combine_videos, get_movie_duration, change_frame_rate, extract_audio, combine_audio, replace_audio, extract_sub_video_sentences
from conversions import get_file_without_path
import glob
import os
import numpy as np
import datetime

#Macros to parse DuckSoup name, change this to correspondent position if needed
date_position_in_file_name = 3
hour_position_in_file_name = 4

##-- trim_folder
def trim_folder(source_folder, folder_tag, target_folder, trimed_path, extension=".mp4", verbose=True):
    #Compute the delays in the videos
    if verbose:
        print("Computing video deltas with ffmpeg...")

    d = {}
    for file in glob.glob(source_folder + "*" + extension):
        if verbose:
            print(file)
            print()

        file_tag = get_file_without_path(file, with_extension=True)
        date = file_tag.split("-")[date_position_in_file_name]
        hour = file_tag.split("-")[hour_position_in_file_name]
        val = datetime.datetime.strptime(date + " " + hour, "%Y%m%d %H%M%S.%f")
        d[file_tag] = val

    if verbose:
        print(d)
        print()

    max_time = max(d.values())
    for k in d.keys():
        val = d[k] - max_time
        val = abs(val.total_seconds())
        d[k] = val

    #trim videos and save
    if verbose:
        print("Triming videos with ffmpeg...")

    if not os.path.isdir(target_folder + trimed_path):
        try:
            os.mkdir(target_folder + trimed_path )
        except:
            pass

    for file in glob.glob(source_folder + "*" + extension):
        if verbose:
            print("Triming file : "+ file)

        file_name = get_file_without_path(file, with_extension=True)
        start = d[file_name]
        length = get_movie_duration(file) - start
        start = str(datetime.timedelta(seconds=start))
        if not os.path.isdir(target_folder + trimed_path + folder_tag):
            os.mkdir(target_folder + trimed_path + folder_tag)

        extract_sub_video_sentences(source_name=file, target_name=target_folder + trimed_path + folder_tag + file_name, start=str(start), length=str(length))


## --re_encode_folder 
def re_encode_folder(source_folder, folder_tag, target_folder, re_encode_path, extension=".mp4",verbose=True):
    if verbose:
        print("Re-encoding videos with ffmpeg...")
        print()

    if not os.path.isdir(target_folder + re_encode_path):
        os.mkdir(target_folder + re_encode_path)

    for file in glob.glob(source_folder +"*" + extension):
        if verbose:
            print("re-encoding : "+ file)

        if not os.path.isdir(target_folder + re_encode_path + folder_tag):
            os.mkdir(target_folder+ re_encode_path + folder_tag)
            
        file_tag = get_file_without_path(file)
        output = target_folder+ re_encode_path + folder_tag + file_tag + ".mp4"
        change_frame_rate(source=file, target_fps = 30, output=output)


##-- combine_folder
def combine_folder(source_folder, folder_tag, target_folder, combined_path, combined_with_audio_path,extension= ".mp4", combine_audio_flag=True, verbose=True):
    if verbose:
        print("combining videos with ffmpeg...")
        print()
    
    combined_video_name = folder_tag.split("/")[-2]
        
    #Combine videos
    nb_files = len(glob.glob(source_folder + "*" + extension))
    if nb_files != 4 :
        print("I can only combine 4 videos, nb file is "+ str(nb_files))
        print()
        return
    
    files = glob.glob(source_folder+"*"+ extension)
    
    if not os.path.isdir(target_folder + combined_path):
        os.mkdir(target_folder+ combined_path)
    
    output = target_folder+combined_path + combined_video_name + extension
    combine_videos( tl = files[0], tr = files[1], bl = files[2], br = files[3], output= output)

    #extract and combine audios
    if combine_audio_flag:
        import uuid 
        if not os.path.isdir(target_folder + combined_with_audio_path):
            os.mkdir(target_folder + combined_with_audio_path)

        audios = []
        for cpt1, file in enumerate(files):
            audio_name = str(cpt1) + str(uuid.uuid1()) +"____.wav"
            extract_audio(file, audio_name)
            audios.append(audio_name)

        aux_wav = str(uuid.uuid1()) + ".wav"
        combine_audio(audios, aux_wav)

        #replace audio
        file_tag = get_file_without_path(output)
        replace_audio(output,  aux_wav, target_folder + combined_with_audio_path + file_tag + extension)

        for audio in audios:
            os.remove(audio)
        
        os.remove( aux_wav)

def ds_process(source_folder
                            , folder_tag
                            , target_folder="preproc/"
                            , extension=".mp4"
                            , trimed_path="trimed/"
                            , re_encode_path = "re-encode/"
                            , combine_videos=True
                            , combined_path="combined/"
                            , combined_with_audio_path="with_audio/"
                            , combine_audio_flag=True
                            , verbose=True
                                    ):
    """
    | A fucntion to preprocess recordings made with ducksoup.
    |
    |
    | This function trims, re-encodes, and combines the videos
    | input:
    |   source_folder : folder where the videos to pre process are located (with a slash at the end)
    |   folder_tag    : is the identifier of the interaction, where all processed videos will be saved
    |   target_folder : is the name of the folder where all the processing will be saved. With a slash in the end. Usually prepoc/
    |   extension : the extension of the videos to process
    |   trimed path : the name of the folder that will be created inside target folder
    |   combine_videos : whether the function should combine videos after preprocessing together into one single video with all four videos. 
    |        This is useful to make sure that videos are in sync.
    |    combine_path : the name of the folder to use when combining the files
    |    verbose : whether to print a detailed output of the stages of the preprocessing


    | Usage Example: 
    |
    |from ducksoup_preproc import ds_process
    |import glob
    |for source_folder in glob.glob("videos/*/recordings/"):
    |   folder_tag = source_folder.split("/")[-3] + "/"
    |   ds_process(source_folder=source_folder, folder_tag=folder_tag)


    """
    # create target folder if it doesn't exist, if it exists, pass
    if not os.path.isdir(target_folder):
        try:
            import errno
            os.mkdir(target_folder)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass
    
    if verbose:
        print(folder_tag)
        print()

    #trim videos
    trim_folder(source_folder=source_folder, folder_tag=folder_tag, target_folder=target_folder, trimed_path=trimed_path, extension=extension, verbose=verbose)

    #reencode
    source_folder = target_folder + trimed_path + folder_tag
    re_encode_folder(source_folder=source_folder, folder_tag=folder_tag, target_folder=target_folder, re_encode_path=re_encode_path, extension=extension, verbose=verbose)

    #Combine videos
    if combine_videos:
        source_folder = target_folder + re_encode_path + folder_tag
        combine_folder(source_folder=source_folder
                       , folder_tag=folder_tag
                       , target_folder=target_folder
                       , combined_path=combined_path
                       , combined_with_audio_path=combined_with_audio_path
                       , extension=extension
                       , verbose=verbose
                       , combine_audio_flag=combine_audio_flag
                       )





## Parallel processing functions
def parallelize_function(source_folder, folder_tag_idx =3
                                    , target_folder="preproc/" 
                                    , extension=".mp4"
                                    , trimed_path="trimed/"
                                    , re_encode_path = "re-encode/"
                                    , combine_videos=True
                                    , combined_path="combined/"
                                    , combined_with_audio_path="with_audio/"
                                    , combine_audio_flag=True
                                    , verbose=True):
    
    folder_tag = source_folder.split("/")[folder_tag_idx] + "/"

    ds_process(source_folder = source_folder
                    , folder_tag = folder_tag
                    , target_folder=target_folder
                    , extension=extension
                    , trimed_path=trimed_path
                    , re_encode_path = re_encode_path
                    , combine_videos=combine_videos
                    , combined_path=combined_path
                    , combined_with_audio_path=combined_with_audio_path
                    , combine_audio_flag=combine_audio_flag
                    , verbose=verbose
                    )

def ds_process_parallel(sources , folder_tag_idx =3
                                    , target_folder="preproc/" 
                                    , extension=".mp4"
                                    , trimed_path="trimed/"
                                    , re_encode_path = "re-encode/"
                                    , combine_videos=True
                                    , combined_path="combined/"
                                    , combined_with_audio_path="with_audio/"
                                    , combine_audio_flag=True
                                    , verbose=True
                                    ):
    """
        from ducksoup import ds_process_parallel
        folder_tag_idx : is the place where the * is, in the path. The place where recoridngs are blocked by.
        ds_process_parallel(sources = "videos/latest_30_10_b/test_2/*/recordings/")
        check help(ds_process) for other arguments

    """
    import multiprocessing
    from itertools import repeat
    
    pool_obj = multiprocessing.Pool()
    a_args     = glob.glob(sources)
    
    pool_obj.starmap(parallelize_function, zip(a_args
                                  , repeat(folder_tag_idx)
                                  , repeat(target_folder)
                                  , repeat(extension)
                                  , repeat(trimed_path)
                                  , repeat(re_encode_path)
                                  , repeat(combine_videos)
                                  , repeat(combined_path)
                                  , repeat(combined_with_audio_path)
                                  , repeat(combine_audio_flag)
                                  , repeat(verbose))
                                  )