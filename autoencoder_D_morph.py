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
modelname = "simple_autoencoder_01"

#baseFolder = '/Users/panagiotismichalatos/Desktop/Autoencoder/'
baseFolder = 'D:/ML/Autoencoder/'

imageFile1 = 'D:/ML/Autoencoder/images/test/n01514859_364.JPEG'
imageFile2 = 'D:/ML/Autoencoder/images/test/n01514859_259.JPEG'
outputFolder = baseFolder + 'images/morphed/'
steps = 10

modelSaveFolder = baseFolder + f'models/{modelname}/'
modelSaveFile = os.path.join(modelSaveFolder, 'model_weights.pth')

#select the model to train
#autoencoderModel  = autoencoder.SimpleAutoencoder((image_size,image_size), inputChannels=image_channels)
#autoencoderModel = autoencoder.VGG16Autoencoder((image_size,image_size), latentSize=400, inputChannels=image_channels)
autoencoderModel = autoencoder.VariationalAutoencoder((image_size,image_size), latentSize=120, inputChannels=image_channels)


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

img1 = Image.open(imageFile1)
img2 = Image.open(imageFile2)

if image_channels==1:
    img1 = img1.convert('L')
    img2 = img2.convert('L')
else:
    img1 = img1.convert('RGB')
    img2 = img2.convert('RGB')

img1 = transform(img1)
img2 = transform(img2)

img1 = img1.to(device)
img1 = img1.unsqueeze(0)

img2 = img2.to(device)
img2 = img2.unsqueeze(0)

#encode the images
img1 = autoencoderModel.encode(img1)
img2 = autoencoderModel.encode(img2)

#interpolate between the two images
for i in range(steps):
    img = img1 + (img2-img1)*i/steps
    img = autoencoderModel.decode(img)
    img = img.squeeze(0)
    img = img.cpu()
    img = img.detach().numpy()
    img = img.transpose(1,2,0)
    img = (img*255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(os.path.join(outputFolder, f'{i}.jpg'))

