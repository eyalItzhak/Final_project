import tensorflow as tf
import json
import numpy as np
import os
from sklearn import preprocessing
from tensorflow import keras
import sys
import subprocess
MODEL_PATH = "./SavedModel/model"
OPEN_POSE = "./openpose"
INPUT = "Prediction_Input"
OUTPUT = "Prediction_Output"

class Centroid:
    def __init__(self,x,y):
        self.x = x
        self.y = y

class PersonData:
    def __init__(self, pose, centroid_x,centroid_y, probability):
        self.pose = pose
        self.centroid = Centroid(centroid_x,centroid_y)
        self.probability = probability

class PersonDataEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, PersonData):
            return {
                'pose': obj.pose,
                'centroid': {'x': obj.centroid.x, 'y': obj.centroid.y},
                'probability': str(obj.probability)
            }
        elif isinstance(obj, Centroid):
            return {'x': obj.x, 'y': obj.y}
        return json.JSONEncoder.default(self, obj)
    

OpenPoseCommand = "start bin/OpenPoseDemo.exe --image_dir \"{input}\" --write_json \"{input}\" --render_pose 0 --display 0"


def run_openPose_on_image(filename):
    file_path = os.path.join(INPUT,filename)
    os.chdir(OPEN_POSE)
    name = os.path.basename(file_path)
    cmd = OpenPoseCommand.format(
        input=os.path.join("../", INPUT),
        output=os.path.join("../", INPUT)
    )
    print("###comannd ===>   " + cmd + "\n")

    # Run the OpenPose command and capture the output and exit status
    process = subprocess.run(cmd, shell=True, capture_output=True)

    # Check if the process was successful
    if process.returncode == 0:
        print("OpenPose process completed successfully.")
    else:
        print(f"OpenPose process failed with exit code {process.returncode}:")
        print(process.stderr.decode('utf-8'))
    #os.system(cmd)
    os.chdir('..')

# normalize key point vector 
def nornamlize_vector(key_points):
    centroid = np.mean(key_points, axis=0)
    key_points -= centroid
    max_distance = np.max(np.sqrt(np.sum(key_points ** 2, axis=1)))
    key_points /= max_distance
    return key_points.tolist(), centroid


def normalize_json(folder):
    # while(len(os.listdir(folder))<2):
    #     time.sleep(0.05)
    files = os.scandir(folder)
    file = list(filter(lambda x : x.name.endswith(".json"),files))[0]
    file_data = json.load(open(file))
    people = []
    for person in file_data["people"]:  # iterate over all people in json file
        key_points = person["pose_keypoints_2d"]
        probabilites = key_points[2::3]
        coordinates_vector = key_points.copy()
        del coordinates_vector[2::3]
        coordinates_vector = [(coordinates_vector[i], coordinates_vector[i+1]) for i in range(0, len(coordinates_vector), 2)]
        normalized_coordinates, centroid = nornamlize_vector(coordinates_vector)
        result = []
        for i in range(len(probabilites)):
            coordinate = normalized_coordinates[i]
            probability = probabilites[i]
            result.append({"coordinate": coordinate,"probability": probability})
        person["pose_keypoints_2d"] = result
        people.append(PersonData("",int(centroid[0]),int(centroid[1]),0))
    json.dump(file_data, open(os.path.join(INPUT, file.name)+"_normalized.json","w"))
    return people


def preprocess_image(image_path):
    # run the image through openpose
    run_openPose_on_image(image_path)
    return normalize_json(INPUT)
     

def get_position_data(folder):
    files = os.scandir(folder)
    file = list(filter(lambda x : x.name.endswith("_normalized.json"),files))[0]
    file_data = json.load(open(file)) 
    data = []
    for person in file_data["people"]: #iterate over all people in json file
        key_points = person["pose_keypoints_2d"]
        person_data  = []
        for key_point in key_points:
            person_data.append(key_point["coordinate"][0])
            person_data.append(key_point["coordinate"][1])
            person_data.append(key_point["probability"])
        data.append(person_data)
    return data


def restore_pose_by_index(index):
    pass

def use_model(model_path,people,int_to_label):
    model = tf.keras.models.load_model(model_path)
    people_position_data = get_position_data(INPUT)
    for i,person_data in enumerate(people_position_data):
        df = np.array(person_data).reshape(1, 75)
        predictions = model.predict(df)
        probability = max(max(predictions))
        pose_index = np.argmax(predictions)
        people[i].probability = float(probability)
        people[i].pose = int_to_label[str(pose_index)] 

    return people
    
def save_predictions(people,output_folder):
    if(not os.path.exists(output_folder)):
        os.mkdir(output_folder)
    with open(os.path.join(output_folder,"people.json"), 'w') as f:
        json.dump([vars(person_data) for person_data in people], f, indent=4,cls=PersonDataEncoder)   

def clean_up_files(input_folder):
    file_list = os.listdir(input_folder)
    for file_name in file_list:
        file_path = os.path.join(input_folder, file_name)  # Get the full path to the file
        os.remove(file_path)  # Delete the file

if __name__ == '__main__':
    # check if the program was run with the correct number of arguments
    if len(sys.argv) != 2:
        print("Usage: python program_name.py input_string")
        sys.exit(1)

    image_path = sys.argv[1]
# Load label-to-index and index-to-label mappings from file
    with open('mappings.json', 'r') as f:
        mappings = json.load(f)
    label_to_int = mappings['label_to_int']
    int_to_label = mappings['int_to_label']
    if(len(os.listdir(INPUT))>0):
        people = preprocess_image(image_path)
        use_model(MODEL_PATH,people,int_to_label)
        save_predictions(people,OUTPUT)
        clean_up_files(INPUT)
    
