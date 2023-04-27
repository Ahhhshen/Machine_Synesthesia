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


#model
model = models.alexnet(weights = models.AlexNet_Weights.DEFAULT)
#model = models.regnet_x_8gf(weights = models.RegNet_X_8GF_Weights.DEFAULT)
#model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
#model = models.mobilenet_v3_small(weights = models.MobileNet_V3_Small_Weights.DEFAULT)
#model = models.vgg16_bn(weights = models.VGG16_BN_Weights.DEFAULT)

#_____________________________optimization parameters
output_folder = 'images/alexenet_735'

target_classes = {2:1.0}

iterations = 500
save_every = 25
learning_rate = 55.0
l2_reg = 0.0

max_jitter = 16
process_every = 5
blur_kernel = 3
gray_strength = 0.0

mseFromOriginalFactor = 0.0
mseFromOriginalHistogram = 0.0
startImageFile = "D:/ML/Autoencoder/images/test/n03891251_97.JPEG"


#_______________________________________Main program starts here
#model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

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
histoLoss = nn.MSELoss()

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

    #canny edge detection
    #gray_1 = cv2.Canny(gray_1, 100, 200)
    #thicken edges
    #kernel = np.ones((3,3),np.uint8)
    #gray_1 = cv2.dilate(gray_1,kernel,iterations = 1)

    #invert
    #gray_1 = 255 - gray_1

    #equalize histogram
    #gray_1 = cv2.equalizeHist(gray_1)

    #binarize image
    #ret, gray_1 = cv2.threshold(gray_1, 127, 255, cv2.THRESH_BINARY)


    gray_3 = cv2.cvtColor(gray_1, cv2.COLOR_GRAY2BGR)
    img = cv2.addWeighted(img, 1.0-gray_strength, gray_3, gray_strength, 0)

    #pixelate image
    #img = cv2.resize(img, (16,16), interpolation=cv2.INTER_AREA)
    #img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
    
    #convert to tensor and normalize and copy to img
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

    if mseFromOriginalHistogram > 0.0:
        #compute histogram of the red channel of the image using torch.histc
        red_histo = torch.histc(img[:,0,:,:], bins=256, min=0, max=1)
        red_histo = red_histo #/ torch.sum(red_histo)

        green_histo = torch.histc(img[:,1,:,:], bins=256, min=0, max=1)
        green_histo = green_histo #/ torch.sum(green_histo)

        blue_histo = torch.histc(img[:,2,:,:], bins=256, min=0, max=1)
        blue_histo = blue_histo #/ torch.sum(blue_histo)

        #compute histogram of the red channel of the original image using torch.histc
        red_histo0 = torch.histc(img0[:,0,:,:], bins=256, min=0, max=1)
        red_histo0 = red_histo0 #/ torch.sum(red_histo0)

        green_histo0 = torch.histc(img0[:,1,:,:], bins=256, min=0, max=1)
        green_histo0 = green_histo0 #/ torch.sum(green_histo0)

        blue_histo0 = torch.histc(img0[:,2,:,:], bins=256, min=0, max=1)
        blue_histo0 = blue_histo0 #/ torch.sum(blue_histo0)

        #compute the histogram loss
        histoLossValue = histoLoss(red_histo, red_histo0) + histoLoss(green_histo, green_histo0) + histoLoss(blue_histo, blue_histo0)
        target_score -= mseFromOriginalHistogram * histoLossValue

    #compute the gradient of the target score with respect to the image
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