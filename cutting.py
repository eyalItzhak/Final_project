from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from os import listdir
import os
import pandas as pd
import pathlib
from os.path import isfile, join


# The format is ('Name of file', start(in seconds), end(in seconds), targetname for the output)
# Example:
# ffmpeg_extract_subclip("Roger Gracie vs Buchecha - 2017 Gracie Pro Jiu-Jitsu.mp4", 82, 102, targetname="test.mp4")

###########################
#Config:
DIRNAME="Input\\1"

 
TARGET_OUTPUT= "Input\\1\\Output"

INFOTYPE1=".csv"
INFOTYPE2=".xlsx"

VIDEOTYPE1=".mp4"
VIDEOTYPE2=".avi"
###########################

global occurance 
occurance= {}

def get_all_files(dir):  # get all files in the folder, return list of csv files and list of video files
    path = pathlib.Path(dir).resolve()
    all_files = listdir(path)
    csv_files = []
    videos = []
    for file in all_files:
        if file.endswith(INFOTYPE1) or file.endswith(INFOTYPE2):
            csv_files.append(str(path) + "\\" + file)
            print("fount data")
        if file.endswith(VIDEOTYPE1) or file.endswith(VIDEOTYPE2):
            videos.append(str(path) + "\\" + file)
            print("fount video")
    # csv_files = [str(path) + "\\" + file for file in all_files if file.endswith(".csv") or file.endswith(".xlsx")]
    return csv_files, videos


def preprocess_all_files(dir,target_output):
    csv_files, videos = get_all_files(dir)
    for csv_file in csv_files:
        csv_file_name = os.path.basename(csv_file)
        if csv_file.__contains__("~"):
            continue
        if (csv_file.endswith(INFOTYPE1)):
            dfs = pd.read_csv(csv_file)
        else:
            dfs = pd.read_excel(csv_file, sheet_name=None)
        print(dfs)
        for key in dfs.keys():  # iterate over all sheets in the exel and convert the time format
            convert_to_seconds_format(dfs[key])
            print(dfs[key])
            for video in videos:
                video_name = os.path.basename(video)
                video_name = video_name.replace(VIDEOTYPE1, '')
                if video_name == csv_file_name or video_name == key:
                   
                    cut_video(video, dfs[key], target_output) #video_file, data, target_output


def convert_to_seconds_format(data):
    start_time = data.loc[:, "From(MM:SS)"]
    for i, time in enumerate(start_time):
        data.loc[:, "From(MM:SS)"][i] = parse_time_format(time)
        data.loc[:, "To(MM:SS)"][i] = parse_time_format(data.loc[:, "To(MM:SS)"][i])


def parse_time_format(time):
    string_time = str(time)
    if not string_time.__contains__(":"):
        return time
    sections = string_time.split(":")
    minutes = int(sections[0])
    seconds = int(sections[1])
    return minutes * 60 + seconds


def cut_video(video_file, data, target_output):
   
    if not os.path.isdir(target_output):
        os.mkdir(target_output)  # create directory for outputs if does not exists
    start_times = data.loc[:, "From(MM:SS)"]
    end_times = data.loc[:, "To(MM:SS)"]
    techniques = data.loc[:, "Technique(move)"]

    for i in range(len(start_times)):
       if techniques[i] in occurance :
          num=occurance[techniques[i]];
          num_occurance = "_"+str(num)
          occurance.update({techniques[i]:num+1})
       else:
          num_occurance="_"+"1"  
          occurance.update({techniques[i]:2})
          
       ffmpeg_extract_subclip(video_file, start_times[i], end_times[i],
                               targetname=target_output + "\\" + str(techniques[i])+num_occurance+".mp4")


# preprocess_all_files(DIRNAME,TARGET_OUTPUT)
