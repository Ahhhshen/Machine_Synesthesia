
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

#image transforms
img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]

def saveImageTensor(img_tensor, img_path, reverseImageNormalization = True):
    img = img_tensor.detach().cpu().numpy()
    if len(img.shape) == 4:
        img = img[0]
    
    channels = img.shape[0]
    #reverse normalization

    if  reverseImageNormalization:
        img[0] = img[0] * img_std[0] + img_mean[0]
        img[1] = img[1] * img_std[1] + img_mean[1]
        img[2] = img[2] * img_std[2] + img_mean[2]

    img = img.transpose(1, 2, 0)
    img = img.clip(0, 1)
    img = img * 255
    img = img.astype('uint8')
    img = Image.fromarray(img)
    img.save(img_path)

def loadImageTensor(img_path, channels:int, device, *, size = None, normalizeImage = True, requires_grad = False):
    img = Image.open(img_path)
    if channels == 1:
        img = img.convert('L')
    else:
        img = img.convert('RGB')

    if size is not None:
        img = img.resize(size, Image.Resampling.BICUBIC)

    img = transforms.ToTensor()(img)
    if normalizeImage:
        img = transforms.Normalize(mean=img_mean, std=img_std)(img)
    return img.clone().detach().to(device).requires_grad_(requires_grad) # torch.tensor(img, requires_grad=requires_grad, device=device)
 
def normalizeImageTensor(img_tensor):
    return transforms.Normalize(mean=img_mean, std=img_std)(img_tensor)

def imageTensorToOpenCV(img_tensor):
    img = img_tensor.detach().cpu().numpy()
    img = img[0]
    #reverse normalization
    
    img[0] = img[0] * img_std[0] + img_mean[0]
    img[1] = img[1] * img_std[1] + img_mean[1]
    img[2] = img[2] * img_std[2] + img_mean[2]

    img = img.transpose(1, 2, 0)
    img = img.clip(0, 1)
    img = img * 255
    img = img.astype('uint8')
    return img

def imageOpenCVToTensor(img, device):
    img = torch.from_numpy(img)
    img = img.permute(2, 0, 1).float() / 255
    img = transforms.Normalize(mean=img_mean, std=img_std)(img)
    img = img.unsqueeze(0).to(device)
    return img

def jitter(X, ox, oy):
    if ox != 0:
        left = X[:, :, :, :-ox]
        right = X[:, :, :, -ox:]
        X = torch.cat([right, left], dim=3)
    if oy != 0:
        top = X[:, :, :-oy]
        bottom = X[:, :, -oy:]
        X = torch.cat([bottom, top], dim=2)
    return X