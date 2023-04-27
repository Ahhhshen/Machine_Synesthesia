import os

targetFolder = "images/audio/train/"
#comnbine this file's path with the target folder
targetFolder = os.path.join(os.path.dirname(__file__), targetFolder)

sampleRate = 44100
duration = 2
numClasses = 3
numSamples = sampleRate * duration
numFiles = 10


# create a dummy dataset of 10 csv files with random waveforms in 3 classes (3 folders)
# each csv file represents a single waveform of 2sec sampled at 44100Hz
# each csv file has 44100*2 = 88200 columns
# each csv file has 1 row

# create 3 folders
# create 10 csv files in each folder

#ensure the target folder exists
if not os.path.exists(targetFolder):
    os.makedirs(targetFolder)

# create 3 folders
for i in range(numClasses):
    #create the folder
    folder = os.path.join(targetFolder, str(i))
    if not os.path.exists(folder):
        os.makedirs(folder)

# create 10 csv files in each folder
import numpy as np
import pandas as pd

for i in range(numClasses):
    for j in range(numFiles):
        #create the csv file
        csvFile = os.path.join(targetFolder, str(i), str(j) + ".csv")
        #create the random waveform
        waveform = np.random.rand(1, numSamples)*2 - 1
        #save the waveform as a csv file
        pd.DataFrame(waveform).to_csv(csvFile, header=False, index=False)




