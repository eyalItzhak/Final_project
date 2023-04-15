import subprocess
import os
import pathlib

REMOVE_DIR_FILE_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(),"../remove_dir_script.py")
PREPROCESS_TRAIN_DATA_FILE_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(),"../Key_point_vector_normalizer.py")
TRAIN_MODEL_FILE_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(),"../Model/NN_For_Keypoints.py")

def run_python_file(file_path):
    try:
        # Run the Python file using subprocess
        result = subprocess.run(['python', file_path], capture_output=True, text=True)
        # Check for any error or return code
        if result.returncode == 0:
            # If return code is 0, print the output
            print("Output:")
            print(result.stdout)
        else:
            # If return code is non-zero, print the error message
            print("Error:")
            print(result.stderr)
    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    # Execute the Python files
    print("Starting to run train and build model pipeline...")

    print("Removing old folders...")
    run_python_file(REMOVE_DIR_FILE_PATH)
    print("Removing old folders completed!")

    print("Preprocessing train input data...")
    run_python_file(PREPROCESS_TRAIN_DATA_FILE_PATH)
    print("Preprocessing train input data completed!")

    print("Training model...")
    run_python_file(TRAIN_MODEL_FILE_PATH)
    print("Training model completed!")
