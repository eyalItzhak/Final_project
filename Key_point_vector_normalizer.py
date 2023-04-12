import json
import numpy as np
import os

INPUT_TRAIN = "Yoga_output_train"
INPUT_TEST = "Yoga_output_test"
MIN_SAMPLE_SIZE = 140 #this number includes the images and the jsons so the real number of samples is half => 70
catagories = []

# normalize key point vector 
def nornamlize_vector(key_points):
    centroid = np.mean(key_points, axis=0)
    key_points -= centroid
    max_distance = np.max(np.sqrt(np.sum(key_points ** 2, axis=1)))
    key_points /= max_distance
    return key_points.tolist()

def convert_json_to_normalize_form(folder):
    files = os.scandir(folder)
    for file in filter(lambda x : x.name.endswith(".json"),files):
            file_data = json.load(open(file)) 
            for person in file_data["people"]: #iterate over all people in json file
                key_points = person["pose_keypoints_2d"]
                probabilites = key_points[2::3]
                coordinates_vector = key_points.copy()
                del coordinates_vector[2::3]
                coordinates_vector = [(coordinates_vector[i],coordinates_vector[i+1]) for i in range(0,len(coordinates_vector),2)]
                normalized_coordinates = nornamlize_vector(coordinates_vector) 
                result = []
                for i in range(len(probabilites)):
                    coordinate = normalized_coordinates[i]
                    probability = probabilites[i]
                    result.append({"coordinate":coordinate,"probability": probability})
                person["pose_keypoints_2d"] = result
            
            output_location = os.path.join(folder,"Normalized_JSON")
            if(not os.path.exists(output_location)):
                os.mkdir(output_location)
            json.dump(file_data,open(os.path.join(output_location,file.name)+"_normalized.json","w"))

def processJsonFile(folder,isTest = False):
    #name = file.name
    files = os.scandir(folder)
    files_list = list(files)
    folder_name = folder.name
    if not isTest and len(files_list) >=MIN_SAMPLE_SIZE:
        convert_json_to_normalize_form(folder)
        catagories.append(folder_name)
    if isTest and folder_name in catagories:
        convert_json_to_normalize_form(folder)
        


def processFiles(base_folder,isTest = False):
    for foldername in os.scandir(base_folder):
      processJsonFile(foldername,isTest)
    
print("Processing train data...")
processFiles(INPUT_TRAIN)
print(len(catagories))
print(catagories)
print("Processing test data...")
processFiles(INPUT_TEST,True)