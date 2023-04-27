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
from z_DDraw import *
from classifier import *

#_______________________________set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#_______________________________________________________________________________________
#_______________________________________________________________________________________
#_______________________________________________________________________________________
#_______________________________________________________________________________________
#_______________________________________________________________________________________
#_______________________________________________________________________________________
#_______________________________________________________________________________________
#EDIT BELOW HERE

#output folder for images. This is where the generated images will be saved
#this folder will be placed under the same folder as the script and it will be created if it doesn't exist
output_folder = 'images/customClassifier8'

#mode of study. Are we trying to match a target image or a target class?
matchTargetImage = False

#if mathcing one or more classes (matchTargetImage=False) then this is the list of classes to match
target_classes = {1:1.0}

#if matching a target image (matchTargetImage=True) then this is the path to the image
startImageFile = "D:/ML/Autoencoder/images/test/n01443537_317.JPEG"

#optimization parameters
#number of iterations
iterations = 1500
#how often to save an image while iterating
save_every = 10

#how fast the parameters of the model change.
#higher vlaues will converge faster but will also be more prone to overshooting and instability
#you need to adjust this rate for each model and target by trial and error
learning_rate = 0.01

#_______________________________model setup
#use a custom classifier you trained previously or one of the pretrained models
useCustomClassifier = True

if useCustomClassifier: #custom classifier parameters if useCustomClassifier=True
    #baseFolder = 'D:/ML/Autoencoder/'
    baseFolder = '/Users/mfchao/Desktop/MachineAesthetics/classifierDir/'
    imageFolder = baseFolder + 'images/train_classes/'

    modelname = "classifier_alexnet"
    #pick model type  [0:"resnet", 1:"alexnet", 2:"vgg", 3:"squeezenet", 4:"densenet"]
    modelType = 1

    classifierModels = ["resnet", "alexnet", "vgg", "squeezenet", "densenet"]
    classifierModel = classifierModels[modelType]

    print(f'Using classifier model: {classifierModel}')

    modelSaveFolder = baseFolder + f'models/{modelname}/'
    modelSaveFile = os.path.join(modelSaveFolder, 'model_weights.pth')

    classFolders = os.listdir(imageFolder)
    classNames = [os.path.splitext(x)[0] for x in classFolders]
    classNames = [x for x in classNames if x != '.DS_Store']
    #sort the class names so that the order is always the same
    classNames.sort()

    model = createClassifierModel(classifierModel, len(classNames), freeze_pretrained_parameters = True, use_pretrained = False)
    model.load_state_dict(torch.load(modelSaveFile))
    model.to(device)    
else: #use pretrained model
    model = models.alexnet(weights = models.AlexNet_Weights.DEFAULT)
    #model = models.regnet_x_8gf(weights = models.RegNet_X_8GF_Weights.DEFAULT)
    #model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
    #model = models.mobilenet_v3_small(weights = models.MobileNet_V3_Small_Weights.DEFAULT)
    #model = models.vgg16_bn(weights = models.VGG16_BN_Weights.DEFAULT)
    #model = models.vit_b_32(weights = models.ViT_B_32_Weights.DEFAULT)

#_______________________________drawing setup
#create drawing
#set the background color
#each rgb component is a number between 0 and 1 or a tuple of (value, (min, max)) 
#where value is the initial value and min and max are the min and max values the parameter can take
bgColor = DColor((0.8, (0,1)), (0.8, (0,1)), (0.8, (0,1)), 0.2)


#image layers
#we can add image layers to the drawing as a static background or in order to 
#visualize saliency maps or gradients

#use an image layer
enableImageLayer = False

#is the image layer variable (can each pixel change during optimization)
isImageVariable = True

#how much to blur the image layer. This is a regularizer that helps the model to focus on the shapes
#when visualizing gradients or saliency maps (isImageVariable=True)
imageBlurAmount = 0.9

#is the image layer monochrome
isImageMonochrome = False

#the image file to use. if (isImageVariable=True) then this is the initial image and will evolve during time
imageFile = '/Users/mfchao/Desktop/MachineAesthetics/classifierDir/images/test/IMG_512200.jpg'

#here we can add 0 or more shapes to the drawing

#if true all shapes will have the same color
areShapesSingleColored = False
#the shared color of all shapes if areShapesSingleColored=True
shapeColor = DColor((0.4, (0,1)), (0.4, (0,1)), (0.4, (0,1)), 0.2)

#a nuber between (0.0 and 0.3) higher values (up to about 0.3) will allow shapes to chose level of bluriness
shapeMaxFuzziness = 0.3

#number of shapes to add
polygonCount = 5
triangleCount = 0
linesCount = 0
circlesCount = 0
curveCount = 0

#_______________________________________________________________________________________
#_______________________________________________________________________________________
#_______________________________________________________________________________________
#_______________________________________________________________________________________
#_______________________________________________________________________________________
#_______________________________________________________________________________________
#_______________________________________________________________________________________
#DO NOT EDIT BELOW HERE
#drawing model construction
drawing = DDrawing(224, 224, device, bgColor)

if enableImageLayer:
    if isImageMonochrome:
        drawing.addMaskedColorLayer(Drgb(0.5, 0.5, 0.5), imageFile, alpha_needs_grad=isImageVariable, post_transforms=transforms.GaussianBlur(3, imageBlurAmount))
    else:
        drawing.addImageLayer(imageFile, color_needs_grad=isImageVariable, post_transforms=transforms.GaussianBlur(3, imageBlurAmount))


#create shape layer
shapeLayer1 = drawing.addSdfGroup([])

for i in range(polygonCount):
    if areShapesSingleColored:
        color = shapeColor
    else:
        color = DColor.random(1.0)

    numPoints = 8 #random.randint(3, 10)
    points = []
    #points are on a circle
    r = 0.3 - i*0.1
    for j in range(numPoints):
        angle = j * 2 * math.pi / numPoints
        x = 0.5 + r * math.cos(angle)
        y = 0.5 + r * math.sin(angle)
        points.append(DPoint((x, (0,1)), (y, (0,1))))
        #points.append(DPoint.random())
    if shapeMaxFuzziness<=0.01:
        falloff = 0.01
    else:
        falloff = DPar.random(0.01, shapeMaxFuzziness)
    padding = 0.2
    sdf = DsdfPolygon(points)
    sdfShape = DsdfShape(sdf, color, falloff, padding)
    shapeLayer1.add(sdfShape)


for i in range(triangleCount):
    if areShapesSingleColored:
        color = shapeColor
    else:
        color = DColor.random(1.0)

    p1 = DPoint.random()
    p2 = DPoint.random()
    p3 = DPoint.random()
    if shapeMaxFuzziness<=0.01:
        falloff = 0.01
    else:
        falloff = DPar.random(0.01, shapeMaxFuzziness)
    padding = 0.3
    sdf = DsdfTriangle(p1, p2, p3)
    sdfShape = DsdfShape(sdf, color, falloff, padding)
    shapeLayer1.add(sdfShape)

for i in range(linesCount):
    if areShapesSingleColored:
        color = shapeColor
    else:
        color = DColor.random(1.0)
    start = DPoint.random()
    end = DPoint.random()
    thickness = DPar.random(0.01, 0.1)
    if shapeMaxFuzziness<=0.01:
        falloff = 0.01
    else:
        falloff = DPar.random(0.01, shapeMaxFuzziness)

    shapeLayer1.addLine(start, end, color, thickness, falloff)


for i in range(circlesCount):
    if areShapesSingleColored:
        color = shapeColor
    else:
        color = DColor.random(1.0)
    center = DPoint.random()
    radius = DPar.random(0.01, 0.25)
    if shapeMaxFuzziness<=0.01:
        falloff = 0.01
    else:
        falloff = DPar.random(0.01, shapeMaxFuzziness)
    padding = 0.3
    sdf = DsdfCircle(center, radius)
    sdfShape = DsdfShape(sdf, color, falloff, padding)
    shapeLayer1.add(sdfShape)


for i in range(curveCount):
    if areShapesSingleColored:
        color = shapeColor
    else:
        color = DColor.random(1.0)
    p1 = DPoint.random()
    p2 = DPoint.random()
    p3 = DPoint.random()

    thickness = DPar.random(0.01, 0.1)
    if shapeMaxFuzziness<=0.01:
        falloff = 0.01
    else:
        falloff = DPar.random(0.01, shapeMaxFuzziness)
    padding = 0.3
    sdf = DsdfBezier2(p1, p2, p3, thickness)
    sdfShape = DsdfShape(sdf, color, falloff, padding)
    shapeLayer1.add(sdfShape)

#print device
print(f'Using device: {device}')

#_______________________________setup output folder
#get this files' directory
dir_path = os.path.dirname(os.path.realpath(__file__))
output_folder = os.path.join(dir_path, output_folder)
#ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#________________________________internal parameters
softmax_factor = 0.0
l2_reg = 0.0

#________________________________model setup
if useCustomClassifier:
    classifierModels = ["resnet", "alexnet", "vgg", "squeezenet", "densenet"]
    classifierModel = classifierModels[modelType]

    print(f'Using classifier model: {classifierModel}')

    modelSaveFolder = baseFolder + f'models/{modelname}/'
    modelSaveFile = os.path.join(modelSaveFolder, 'model_weights.pth')

    image_size = 224
    image_channels = 3
    freeze_pretrained_parameters = True

    classFolders = os.listdir(imageFolder)
    classNames = [os.path.splitext(x)[0] for x in classFolders]
    classNames = [x for x in classNames if x != '.DS_Store']
    #sort the class names so that the order is always the same
    classNames.sort()

    print(f'Found {len(classNames)} classes: {classNames}')

    # Create the model
    model = createClassifierModel(classifierModel, len(classNames), freeze_pretrained_parameters = freeze_pretrained_parameters, use_pretrained = False)
    model.load_state_dict(torch.load(modelSaveFile))


model = model.to(device)


#________________________________TEMP DRAWING TESTS

#imgfolder = "D:/ML/Autoencoder/images/test/"
#image1 = drawing.addImageLayer(imgfolder + "n01443537_374.JPEG")
#blur = transforms.GaussianBlur(3, 0.5)

#image1 = drawing.addImageLayer(imgfolder + "n01443537_374.JPEG", color_needs_grad=True, post_transforms=blur)
#image1 = drawing.addImageLayer(color_needs_grad=True, post_transforms=blur)
#mask1 = drawing.addMaskedColorLayer(Drgb(0.5, 0.5, 0.5), imgfolder + "n01514859_259.JPEG", alpha_needs_grad=True, post_transforms=blur)
#mask1 = drawing.addMaskedColorLayer(Drgb(0.0, 0.0, 0.0), alpha_value=0.5, alpha_needs_grad=True)

#create shapes
# color1 = Drgb(0.0, 0.4, 0.9)
# center1 = DPoint((0.5, (0,1)), (0.5, (0,1)))
# radius1 = DPar(50.0, (0.1, 100.0))
# strength1 = DPar(1.0, (0.1, 1))
# blob1 = DBlob(center1, radius1, strength1, color1)


#create shape group
# group1 = drawing.addShapeGroup([blob1])

# #add 10 random blobs
# for i in range(15):
#     color = DColor.random(1.0)
#     center = DPoint.random()
#     radius = DPar.random(0.1, 20.0)
#     strength = DPar.random(0.1, 1.0)
#     blob = DBlob(center, radius, strength, color)
#     group1.add(blob)



#add 10 random sdf circles
# for i in range(15):
#     color = DColor.random(1.0)
#     center = DPoint.random()
#     radius = DPar.random(0.01, 0.25)
#     falloff = DPar.random(0.01, 0.1)
#     padding = 0.3
#     sdf = DsdfCircle(center, radius)
#     sdfShape = DsdfShape(sdf, color, falloff, padding)
#     sdfgroup.add(sdfShape)

#add 10 random sdf lines
# for i in range(8):
#     color = DColor.random(1.0)
#     start = DPoint.random()
#     end = DPoint.random()
#     radius = DPar.random(0.01, 0.1)
#     falloff = 0.01 # DPar.random(0.01, 0.1)
#     padding = 0.3
#     sdf = DsdfLine(start, end, radius)
#     sdfShape = DsdfShape(sdf, color, falloff, padding)
#     sdfgroup.add(sdfShape)


#add 10 random sdf triangles
# for i in range(10):
#     color = DColor.random(0.9)
#     p1 = DPoint.random()
#     p2 = DPoint.random()
#     p3 = DPoint.random()
#     falloff = 0.01 # DPar.random(0.01, 0.1)
#     padding = 0.3
#     sdf = DsdfTriangle(p1, p2, p3)
#     sdfShape = DsdfShape(sdf, color, falloff, padding)
#     sdfgroup.add(sdfShape)

#add 10 random sdf bezier curves
# for i in range(10):
#     color = DColor.random(1.0)
#     p1 = DPoint.random()
#     p2 = DPoint.random()
#     p3 = DPoint.random()

#     thickness = DPar.random(0.01, 0.1)
#     falloff = 0.01 # DPar.random(0.01, 0.1)
#     padding = 0.3
#     sdf = DsdfBezier2(p1, p2, p3, thickness)
#     sdfShape = DsdfShape(sdf, color, falloff, padding)
#     sdfgroup.add(sdfShape)


#add 3 random sdf polygons
# for i in range(2):
#     color = DColor.random(1.0)
#     numPoints = 16 #random.randint(3, 10)
#     points = []
#     #points are on a circle
#     r = 0.3 - i*0.1
#     for j in range(numPoints):
#         angle = j * 2 * math.pi / numPoints
#         x = 0.5 + r * math.cos(angle)
#         y = 0.5 + r * math.sin(angle)
#         points.append(DPoint((x, (0,1)), (y, (0,1))))
#         #points.append(DPoint.random())
#     falloff = 0.05 # DPar.random(0.01, 0.1)
#     padding = 0.2
#     sdf = DsdfPolygon(points)
#     sdfShape = DsdfShape(sdf, color, falloff, padding)
#     sdfgroup.add(sdfShape)

#create parameters
drawing.initParameters(randomize=False)

if matchTargetImage:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=img_utils.img_mean, std=img_utils.img_std)
    ])

    
    img0 = Image.open(startImageFile).convert('RGB') 
    img0 = transform(img0).unsqueeze(0).to(device)   
    img0 = torch.tensor(img0, requires_grad=False, device=device)

    #create mse loss function
    mse = nn.MSELoss()

#create adam optimizer
optimizer = torch.optim.Adam(drawing.getOptimizableParameters(), lr=learning_rate)#, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

#main optimization loop
for i in range(iterations):

    optimizer.zero_grad()         

    img = drawing.draw()
    img = img.unsqueeze(0) 

    if matchTargetImage:
        loss = mse(img, img0)  - l2_reg * torch.norm(img)
    else:
        #ox, oy = random.randint(0, 16), random.randint(0, 16)
        #img.data.copy_(img_utils.jitter(img.data, ox, oy))

        model.zero_grad()
        #apply normalization to image
        img = img_utils.normalizeImageTensor(img)
        
        score = model(img)
        loss = 0.0 * score[:, 0] - l2_reg * torch.norm(img)

        if softmax_factor>0:
            soft_score = torch.nn.functional.softmax(score, dim=1)* softmax_factor
            for target_class, weight in target_classes.items():
                loss -= weight * (score[:, target_class] + soft_score[:, target_class])
        else:
            for target_class, weight in target_classes.items():
                loss -= weight * score[:, target_class]

        
    loss.backward()
    optimizer.step()
    drawing.postStep(use_no_grad=False)

    #if not matchTargetImage:
        #img.data.copy_(img_utils.jitter(img.data, -ox, -oy))

    if i % save_every == 0:                 
        iter_image = os.path.join(output_folder, f'iter_{i}.png')
        img_utils.saveImageTensor(img, iter_image, reverseImageNormalization = not matchTargetImage)
        print(f'Iteration: {i} / {iterations} : target score: {loss.item()}')
        #print(drawing.params.tensor)