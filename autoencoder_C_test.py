import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import autoencoder
import numpy as np
import os
from PIL import Image

image_size = 64
image_channels = 3
modelname = "Simple_01"

#baseFolder = '/Users/panagiotismichalatos/Desktop/Autoencoder/'
baseFolder = '/Users/panagiotismichalatos/Desktop/Autoencoder/'
#baseFolder = 'D:/ML/Autoencoder/'

imageFolder = baseFolder + 'images/test/'
outputFolder = baseFolder + 'images/recoded/'

modelSaveFolder = baseFolder + f'models/{modelname}/'
modelSaveFile = os.path.join(modelSaveFolder, 'model_weights.pth')

#select the model to train
autoencoderModel  = autoencoder.SimpleAutoencoder((image_size,image_size), inputChannels=image_channels)
#autoencoderModel = autoencoder.VGG16Autoencoder((image_size,image_size), latentSize=400, inputChannels=image_channels)
#autoencoderModel = autoencoder.VariationalAutoencoder((image_size,image_size), latentSize=120, inputChannels=image_channels)


if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)

#check if cuda is available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


#initialize the autoencoder
autoencoderModel.to(device)
autoencoderModel.load_state_dict(torch.load(modelSaveFile))
autoencoderModel.eval()


transform = transforms.Compose([
    transforms.Resize((image_size,image_size)), 
    transforms.ToTensor(),
    autoencoder.normXForm
    ])


#for each image in the imageFolder, recode it and save it in the outputFolder
for filename in os.listdir(imageFolder):
    if filename.endswith('.DS_Store'):
        continue
    
    print(f'processing {filename}')
    img = Image.open(os.path.join(imageFolder, filename))

    if image_channels==1:
        img = img.convert('L')
    else:
        img = img.convert('RGB')

    img = transform(img)
    img = img.to(device)
    img = img.unsqueeze(0)
    output = autoencoderModel(img)
    output = output.squeeze(0)
    output = output.detach().cpu()
    output = autoencoder.normXFormInv(output).clamp(0, 1)
    output = transforms.ToPILImage()(output)
    output.save(os.path.join(outputFolder, filename))
