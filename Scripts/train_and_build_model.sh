#!/bin/bash

# Specify the path to your Python interpreter
PYTHON_INTERPRETER="/usr/bin/python"

# Specify the paths to your Python files
PYTHON_FILE1="../remove_dir_script.py"
PYTHON_FILE2="../Key_point_vector_normalizer.py"
PYTHON_FILE3="../Model/NN_For_Keypoints.py"

# Execute the Python files using the Python interpreter
print("Starting to run train and build model pipeline...")
print("Removing old folders...")
$PYTHON_INTERPRETER $PYTHON_FILE1
print("Removing old folders completed!")

print("Preprocessing train input data...")
$PYTHON_INTERPRETER $PYTHON_FILE2
print("Preprocessing train input data completed!")

print("Training model...")
$PYTHON_INTERPRETER $PYTHON_FILE3
print("Training model completed!")
