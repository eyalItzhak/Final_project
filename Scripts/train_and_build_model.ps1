# Specify the path to your Python interpreter
$pythonInterpreter = "C:\Users\ransa\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Python 3.7\python3.7.exe"
$currentDirectory = $PSScriptRoot

# Get the root folder of the current script directory
$parentDirectory = Split-Path -Path $currentDirectory -Parent

# Specify the paths to your Python files
$pythonFile1 = $parentDirectory + "\remove_dir_script.py"
$pythonFile2 = $parentDirectory + "\Key_point_vector_normalizer.py"
$pythonFile3 = $parentDirectory+ "\Model\NN_For_Keypoints.py"
Write-Host($pythonFile1)

# Execute the Python files using the Python interpreter
("Starting to run train and build model pipeline...")
Write-Host("Removing old folders...")
Start-Process $pythonInterpreter $pythonFile1
Write-Host("Removing old folders completed!")

Write-Host("Preprocessing train input data...")
Start-Process $pythonInterpreter $pythonFile2
Write-Host("Preprocessing train input data completed!")

Write-Host("Training model...")
Start-Process $pythonInterpreter $pythonFile3
Write-Host("Training model completed!")

Read-Host "Press Enter to exit..."


