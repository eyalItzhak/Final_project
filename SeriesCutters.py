import cutting as ct 
import os
BASE_PATH = "Input"
OUTPUT_PATH= "Output_RAW"


def SeriesCutters ():
   for root, dirs, files in os.walk(BASE_PATH):
        for dir in dirs:
             ct.preprocess_all_files(BASE_PATH+"\\"+dir,OUTPUT_PATH)

        
SeriesCutters();        
