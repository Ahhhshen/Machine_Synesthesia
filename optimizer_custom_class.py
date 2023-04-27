import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import os
import torchvision.transforms as transforms
import random
import cv2
import numpy as np
import optimizer_image_utils as img_utils
from classifier import *

#________________________________classifier parameters
modelname = "classifier_alexnet"
#pick model type  [0:"resnet", 1:"alexnet", 2:"vgg", 3:"squeezenet", 4:"densenet"]
modelType = 1

baseFolder = 'D:/ML/Autoencoder/'
#baseFolder = '/Users/panagiotismichalatos/Desktop/classifiers/'
imageFolder = baseFolder + 'images/train_classes/'

#_____________________________optimization parameters
output_folder = 'images/alexenet_fish_ostrich'

target_classes = {1:1.0}

iterations = 500
save_every = 25
learning_rate = 55.0
l2_reg = 0.0

max_jitter = 16
process_every = 5
blur_kernel = 3
gray_strength = 0.01

mseFromOriginalFactor = 100.0
startImageFile = "D:/ML/Autoencoder/images/test/n03891251_97.JPEG"

#_______________________________________Main program starts here
#model
classifierModels = ["resnet", "alexnet", "vgg", "squeezenet", "densenet"]
classifierModel = classifierModels[modelType]

print(f'Using classifier model: {classifierModel}')

modelSaveFolder = baseFolder + f'models/{modelname}/'
modelSaveFile = os.path.join(modelSaveFolder, 'model_weights.pth')


image_size = 224
image_channels = 3
freeze_pretrained_parameters = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

classFolders = os.listdir(imageFolder)
classNames = [os.path.splitext(x)[0] for x in classFolders]
classNames = [x for x in classNames if x != '.DS_Store']
#sort the class names so that the order is always the same
classNames.sort()


print(f'Found {len(classNames)} classes: {classNames}')

# Create the model
model = createClassifierModel(classifierModel, len(classNames), freeze_pretrained_parameters = freeze_pretrained_parameters, use_pretrained = False)
model.load_state_dict(torch.load(modelSaveFile))
model.to(device)
#classifierModel.eval()

#source image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=img_utils.img_mean, std=img_utils.img_std)
])

img = Image.open(startImageFile).convert('RGB') 
img = transform(img).unsqueeze(0).to(device)   
img = torch.tensor(img, requires_grad=True, device=device)

img0 = Image.open(startImageFile).convert('RGB') 
img0 = transform(img0).unsqueeze(0).to(device)   
img0 = torch.tensor(img0, requires_grad=False, device=device)

mseLoss = nn.MSELoss()


#get this files' directory
dir_path = os.path.dirname(os.path.realpath(__file__))
output_folder = os.path.join(dir_path, output_folder)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def image_process(img_orig):
    img = img_utils.imageTensorToOpenCV(img_orig)
    #blur image
    img = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)

    #reduce saturation
    gray_1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    gray_3 = cv2.cvtColor(gray_1, cv2.COLOR_GRAY2BGR)
    img = cv2.addWeighted(img, 1.0-gray_strength, gray_3, gray_strength, 0)
    img = img_utils.imageOpenCVToTensor(img, device)
    img_orig.data.copy_(img)

#based on  
#https://github.com/chriskhanhtran/CS231n-CV/blob/master/assignment3/NetworkVisualization-PyTorch.ipynb

#main optimization loop
for i in range(iterations):

    # Randomly jitter the image a bit; this gives slightly nicer results
    ox, oy = random.randint(0, max_jitter), random.randint(0, max_jitter)
    img.data.copy_(img_utils.jitter(img.data, ox, oy))

    score = model(img)
    target_score = 0.0 * score[:, 0] - l2_reg * torch.norm(img)
    for target_class, weight in target_classes.items():
        target_score += weight * score[:, target_class]

    # Add the MSE loss from the original image   
    mseLossValue = mseLoss(img, img0)
    target_score -= mseFromOriginalFactor * mseLossValue

    target_score.backward()
    with torch.no_grad():
        img += learning_rate * img.grad / torch.norm(img.grad)
        img.grad.zero_()
    
    # Undo the random jitter
    img.data.copy_(img_utils.jitter(img.data, -ox, -oy))

    # As regularizer, clamp and periodically blur the image
    for c in range(3):
        lo = float(-img_utils.img_mean[c] / img_utils.img_std[c])
        hi = float((1.0 - img_utils.img_mean[c]) / img_utils.img_std[c])
        img.data[:, c].clamp_(min=lo, max=hi)
    if i % process_every == 0:
        image_process(img)

    if i % save_every == 0:        
        iter_image = os.path.join(output_folder, f'iter_{i}.png')
        img_utils.saveImageTensor(img, iter_image)
        print(f'Iteration: {i} / {iterations}')