import os
import subprocess,shlex

OPEN_POSE="openpose"
INPUT = "Datasets/data1/test"
OUTPUT= "Yoga_output"
MAX_PEOPLE="3"

OpenPoseCommand="bin\\OpenPoseDemo.exe --image_dir \"{input}\" --write_images \"{output}\" --write_json \"{output}\""
#ExtrOption ="--number_people_max " + MAX_PEOPLE + " --model_pose BODY_135"

def processFile(filename):
    os.chdir(OPEN_POSE)
    name = filename.name
    cmd = OpenPoseCommand.format(
        input = os.path.join("..\\",INPUT,name),
        output = os.path.join("..\\",OUTPUT,name)
        )
    # subprocess.run(shlex.split(cmd))
    #cmd= cmd + ExtrOption
    print("###comannd ===>   " + cmd + "\n")
    os.system(cmd)
    os.chdir('..')

def processFiles():
    for folder in os.scandir(INPUT):
      print(folder.path) #for debug
      processFile(folder)



processFiles();     


