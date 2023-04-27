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

baseFolder = '/Users/panagiotismichalatos/Desktop/Autoencoder/'
#baseFolder = 'D:/ML/Autoencoder/'

imageFolder = baseFolder + 'images/test/'
csvFile = 'autoencoder_embedding.csv'

modelSaveFolder = baseFolder + f'models/{modelname}/'
modelSaveFile = os.path.join(modelSaveFolder, 'model_weights.pth')

#select the model to train
autoencoderModel  = autoencoder.SimpleAutoencoder((image_size,image_size), inputChannels=image_channels)
#autoencoderModel = autoencoder.VGG16Autoencoder((image_size,image_size), latentSize=400, inputChannels=image_channels)
#autoencoderModel = autoencoder.VariationalAutoencoder((image_size,image_size), latentSize=120, inputChannels=image_channels)

#set csv file in th ecurrent python file folder
csvFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), csvFile)

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



with open(csvFile, 'w') as f:
    for filename in os.listdir(imageFolder):
        if filename.endswith('.DS_Store'):
            continue
        
        filepath = os.path.join(imageFolder, filename)
        print(f'processing {filepath}')
        img = Image.open(filepath)

        if image_channels==1:
            img = img.convert('L')
        else:
            img = img.convert('RGB')

        img = transform(img)
        img = img.to(device)
        img = img.unsqueeze(0)
        output = autoencoderModel.encode(img).flatten()
        output = output.squeeze(0)
        output = output.detach().cpu().numpy()
        
        #write filename and output to csv file

        f.write(f'{filepath},')
        for i in range(output.shape[0]):
            if i!=0: f.write(',')
            f.write(f'{output[i]}')
        f.write('\n')







