import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
import ssl
import os
import pandas as pd
import random
from audioClassifier import *

    
modelFile = "images/audio/model/audioClassifier.pt"
csvFile = "images/audio/train/1/0.csv"
sampleCount = 32000
numClasses = 3

modelFile = os.path.join(os.path.dirname(__file__), modelFile)
csvFile = os.path.join(os.path.dirname(__file__), csvFile)

#get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AudioClassifier(numClassea=numClasses)
model.load_state_dict(torch.load(modelFile))
model = model.to(device)
model.eval()


#load the csv file
waveform = pd.read_csv(csvFile, header=None).values[0]
if len(waveform) > sampleCount:
    waveform = waveform[:sampleCount]
waveform = torch.from_numpy(waveform).float().to(device)
waveform = waveform.unsqueeze(0).unsqueeze(0)

#predict
output = model(waveform)
predictionVector = output.squeeze().squeeze().detach().cpu().numpy() 
print(predictionVector)


