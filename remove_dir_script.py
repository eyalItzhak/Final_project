import os
import pathlib

TRAIN_INPUT = "Yoga_output_train"
TEST_INPUT = "Yoga_output_test"
def remove(input):
    path = os.path.join(pathlib.Path(__file__).parent.resolve(),input)
    for folder_name in os.scandir(path):
        norm_folder = os.path.join(folder_name,"Normalized_JSON")
        if(os.path.exists(norm_folder)):
            for file in os.scandir(norm_folder):
                os.remove(file)
            os.removedirs(norm_folder)


remove(TRAIN_INPUT)
remove(TEST_INPUT)

