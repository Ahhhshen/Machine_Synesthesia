import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
import ssl
import os
import pandas as pd
import random

#adapted from https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/audio_classifier_tutorial.ipynb#scrollTo=qdcvcHPywBBR


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, trainFolderPath, sampleCount = 32000, preload = True):
        self.trainFolderPath = trainFolderPath
        self.files = []
        self.labels = []
        self.sampleCount = sampleCount
        self.preload = preload
        
        #get folder names filter only directory type files
        self.labels = [f for f in os.listdir(self.trainFolderPath) if os.path.isdir(os.path.join(self.trainFolderPath, f))]
        #sort the folder names
        self.labels.sort()

        #get file names and store as (fullpath, label) tuple
        for i, label in enumerate(self.labels):
            folderPath = os.path.join(self.trainFolderPath, label)
            for file in os.listdir(folderPath):
                if preload:
                    data = torch.tensor(pd.read_csv(os.path.join(folderPath, file), header=None).values, dtype=torch.float32)[0]
                    self.files.append((data, i))
                else:
                    self.files.append((file,i))

    def __getitem__(self, index):
        

        if self.preload:
            sound, labelId = self.files[index]
        else:
            filename, labelId = self.files[index]
            label = self.labels[labelId]
            filePath = os.path.join(self.trainFolderPath, label, filename)

            #load csv file as a tensor
            sound = torch.tensor(pd.read_csv(filePath, header=None).values, dtype=torch.float32)[0]

        soundData = torch.zeros([self.sampleCount]) #tempData accounts for audio clips that are too short
        if sound.numel() < self.sampleCount:            
            soundData[:sound.numel()] = sound[:]
        elif sound.numel() > self.sampleCount:
            #get random subrange
            start = random.randint(0, sound.numel() - self.sampleCount)
            soundData = sound[start:start + self.sampleCount]
        else:
            soundData[:] = sound[:]

        #format soundData to have 1 channel
        soundData = soundData.unsqueeze(0)
        return soundData, labelId
    
    def __len__(self):
        return len(self.files)
    
class AudioClassifier(nn.Module):
    def __init__(self, numClassea):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 128, 80, 4)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, 3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128, 256, 3)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(256, 512, 3)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(30) #input should be 512x30 so this outputs a 512x1
        self.fc1 = nn.Linear(512, numClassea)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = self.avgPool(x)
        x = x.permute(0, 2, 1) #change the 512x1 to 1x512
        x = self.fc1(x)
        return F.log_softmax(x, dim = 2)
    