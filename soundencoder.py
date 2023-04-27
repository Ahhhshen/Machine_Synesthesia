import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from pytorch_msssim import SSIM, MS_SSIM
import torchvision
import numpy as np
from torchvision import datasets, models, transforms
from PIL import Image
import os

normMean = [0.0, 0.0, 0.0] # [0.5, 0.5, 0.5]
normStd =  [1.0, 1.0, 1.0] #[0.25, 0.25, 0.25]



normXForm = transforms.Normalize(normMean, normStd)
normXFormInv = transforms.Normalize([-x / y for x, y in zip(normMean, normStd)], [1.0 / x for x in normStd])


def calcConvolutionShape(h_w, kernel_size=(1,1), stride=(1,1), pad=(0,0), dilation=1):    
    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1
    
    return h, w

class SoundDataset(torch.utils.data.Dataset):
    def __init__(self, imageFolder, image_channels = 3, transform=None):
        self.imageFolder = imageFolder
        self.transform = transform
        self.imageList = os.listdir(imageFolder)

        #remove .DS_Store files
        self.imageList = [x for x in self.imageList if not x.endswith('.DS_Store')]
        
        self.image_channels = image_channels
        
    def __len__(self):
        return len(self.imageList)
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.imageFolder, self.imageList[idx]))
        if self.image_channels == 1:
            img = img.convert('L')
        else:
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img


class SoundEncoder(nn.Module):
    def __init__(self, wavelformSize : int, classCount : int):
        super().__init__()    
        self.wavelformSize = wavelformSize
        self.classCount = classCount
        self.encoder = nn.Sequential(
            nn.Linear(wavelformSize, 1024), 
            nn.ReLU(True),
            nn.Linear(1024, classCount), 
            nn.Sigmoid()
        )

    def initWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0.01)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        

    def printShapes(self, size, device):
        #create a dummy input of size size (e.g. 64x64)
        dummy_input = torch.rand(1, 3, size[0], size[1])
        dummy_input = dummy_input.to(device)
        #print the shape of each layer in the encoder and decoder using the dummy input
        print("Encoder:")
        for name, layer in self.encoder.named_children():
            dummy_input = layer(dummy_input)
            print(name, "[",layer, "]", ":", dummy_input.shape)

        dummy_input = torch.rand(1, 3, size[0], size[1])
        dummy_input = dummy_input.to(device)
        dummy_input = self.encode(dummy_input)
        print("Decoder:")
        for name, layer in self.decoder.named_children():
            dummy_input = layer(dummy_input)
            print(name, "[",layer, "]", ":", dummy_input.shape)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
