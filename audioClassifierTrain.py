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

    
trainFolder = "images/audio/train/"
modelFolder = "images/audio/model/"
modelFileName = "audioClassifier.pt"
#comnbine this file's path with the target folder
trainFolder = os.path.join(os.path.dirname(__file__), trainFolder)
modelFolder = os.path.join(os.path.dirname(__file__), modelFolder)


sampleCount = 32000
batch_size = 10
epochs = 20

#get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#ensure the model folder exists
if not os.path.exists(modelFolder):
    os.makedirs(modelFolder)

dataset = AudioDataset(trainFolder, sampleCount=sampleCount, preload=True)
dataLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

numClasses = len(dataset.labels)
print("Number of classes: " + str(numClasses))

model = AudioClassifier(numClassea=numClasses).to(device)
print(model)
model.train()

#loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay = 0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)

#train the model

for epoch in range(epochs):
    for batchId, (data, target) in enumerate(dataLoader):
        data = data.to(device)
        target = target.to(device)
        
        #data = data.requires_grad_() #set requires_grad to True for training

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output.squeeze(), target)
        loss.backward()
        optimizer.step()
        
        if batchId % 10 == 0:
            print("Epoch: {} Batch: {} Loss: {}".format(epoch, batchId, loss.item()))
    scheduler.step()

#save the model
torch.save(model.state_dict(), os.path.join(modelFolder, "audioClassifier.pt"))