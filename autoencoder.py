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

class ImageDataset(torch.utils.data.Dataset):
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

class AutoencoderBase(nn.Module):
    def __init__(self):
        super().__init__()    
        self.encoder : nn.Module = None
        self.decoder : nn.Module = None 
        self.latentSize : int = 0

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


class SimpleAutoencoder(AutoencoderBase):
    def __init__(self, imgSize, inputChannels = 3, kernelSize = 3, featureChannels = 4, useSSIMLoss = False):
        super().__init__()

        reducedSize1 = calcConvolutionShape(imgSize, (kernelSize,kernelSize), (2,2), (1,1))
        reducedSize2 = calcConvolutionShape(reducedSize1, (kernelSize,kernelSize), (2,2), (1,1))
        reducedSize3 = calcConvolutionShape(reducedSize2, (kernelSize,kernelSize), (2,2), (1,1))

        self.latentSize = featureChannels*reducedSize3[0]*reducedSize3[1]

        self.useSSIMLoss = useSSIMLoss

        if self.useSSIMLoss:
            self.criterion = SSIM(win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=3)
            #criterion = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=3)
        else:
            self.criterion = nn.MSELoss()

        self.encoder = nn.Sequential(
            nn.Conv2d(inputChannels, 8, kernelSize, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernelSize, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(16, featureChannels, kernelSize, stride=2, padding=1),
            nn.ReLU(True)
        )
            
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(featureChannels, 16, kernelSize, stride=2, padding=1, output_padding=1),  
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 8, kernelSize, stride=2, padding=1, output_padding=1),  
                nn.ReLU(True),
                nn.ConvTranspose2d(8, inputChannels, kernelSize, stride=2, padding=1, output_padding=1), 
                #nn.Tanh()
                nn.Sigmoid()
            )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def computeLoss(self, x, xHat):
        if self.useSSIMLoss:
            return 1 -  self.criterion(xHat, x)
        else:
            return  self.criterion(xHat, x)
    


#https://blog.paperspace.com/convolutional-autoencoder/
class VGG16Autoencoder(AutoencoderBase):
    def __init__(self, imgSize, latentSize = 50, inputChannels = 3, outputChannels = 16, useSSIMLoss = False):
        super().__init__()

        reducedSize1 = calcConvolutionShape(imgSize, (3,3), (2,2), (1,1))
        reducedSize2 = calcConvolutionShape(reducedSize1, (3,3), (2,2), (1,1))

        self.latentSize = latentSize
        
        self.useSSIMLoss = useSSIMLoss

        if self.useSSIMLoss:
            self.criterion = SSIM(win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=3)
            #criterion = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=3)
        else:
            self.criterion = nn.MSELoss()

        self.encoder = nn.Sequential(
            nn.Conv2d(inputChannels, outputChannels, 3, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(outputChannels, outputChannels, 3, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(outputChannels, 2*outputChannels, 3, padding=1, stride=2), 
            nn.ReLU(True),
            nn.Conv2d(2*outputChannels, 2*outputChannels, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(2*outputChannels, 4*outputChannels, 3, padding=1, stride=2), 
            nn.ReLU(True),
            nn.Conv2d(4*outputChannels, 4*outputChannels, 3, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(4*outputChannels*reducedSize2[0]*reducedSize2[1], latentSize),
            nn.ReLU(True)
        )
            
        self.decoder = nn.Sequential(
                nn.Linear(latentSize, 4*outputChannels*reducedSize2[0]*reducedSize2[1]),
                nn.ReLU(True),
                nn.Unflatten(1, (4*outputChannels, reducedSize2[0], reducedSize2[1])),
                nn.ConvTranspose2d(4*outputChannels, 4*outputChannels, 3, padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(4*outputChannels, 2*outputChannels, 3, padding=1, stride=2, output_padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(2*outputChannels, 2*outputChannels, 3, padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(2*outputChannels, outputChannels, 3, padding=1, stride=2, output_padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(outputChannels, outputChannels, 3, padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(outputChannels, inputChannels, 3, padding=1),
                nn.Sigmoid()
            )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def computeLoss(self, x, xHat):
        if self.useSSIMLoss:
            return 1 -  self.criterion(xHat, x)
        else:
            return  self.criterion(xHat, x)
            


#https://avandekleut.github.io/vae/
class VariationalAutoencoder(AutoencoderBase):
    def __init__(self, imgSize, latentSize = 50, inputChannels = 3, outputChannels = 16):
        super().__init__()

        reducedSize1 = calcConvolutionShape(imgSize, (3,3), (2,2), (1,1))
        reducedSize2 = calcConvolutionShape(reducedSize1, (3,3), (2,2), (1,1))

        self.latentSize = latentSize

        self.encoder = nn.Sequential(
            nn.Conv2d(inputChannels, outputChannels, 3, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(outputChannels, 2*outputChannels, 3, padding=1, stride=2), 
            nn.ReLU(True),
            nn.Conv2d(2*outputChannels, 4*outputChannels, 3, padding=1, stride=2), 
            nn.ReLU(True),
            nn.Flatten()           
        )

        self.linearToMean = nn.Linear(4*outputChannels*reducedSize2[0]*reducedSize2[1], latentSize)
        self.linearToStd = nn.Linear(4*outputChannels*reducedSize2[0]*reducedSize2[1], latentSize)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0
            
        self.decoder = nn.Sequential(
                nn.Linear(latentSize, 4*outputChannels*reducedSize2[0]*reducedSize2[1]),
                nn.Unflatten(1, (4*outputChannels, reducedSize2[0], reducedSize2[1])),
                nn.ReLU(True),                
                nn.ConvTranspose2d(4*outputChannels, 2*outputChannels, 3, padding=1, stride=2, output_padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(2*outputChannels, outputChannels, 3, padding=1, stride=2, output_padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(outputChannels, inputChannels, 3, padding=1),
                nn.Sigmoid()
            )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def computeLoss(self, x, xHat):
        return ((x - xHat)**2).sum() + self.kl

    def encode(self, x):
        x = self.encoder(x)
        mu = self.linearToMean(x)
        log_std = self.linearToStd(x)
        sigma = log_std.exp()
        self.kl = self.klLoss(mu, log_std, sigma)
        z = self.N.sample(mu.shape).to(mu.device) * sigma + mu
        return z

    def klLoss(self, mu, log_std, sigma):
        return -0.5*(1 + log_std - mu**2 - sigma**2).sum()
        #return (sigma**2 + mu**2 - torch.log(sigma) - 0.5).sum()



