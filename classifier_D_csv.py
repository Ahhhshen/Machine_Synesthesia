import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import numpy as np
import os
from PIL import Image
from classifier import *

#set the parameters for the model training
modelname = "classifier_alexnet"
#pick model type  [0:"resnet", 1:"alexnet", 2:"vgg", 3:"squeezenet", 4:"densenet"]
modelType = 1

baseFolder = 'D:/OneDrive/Harvard/OneDrive - Harvard University/SCI6487_machine_aesthetics/Final_projects/Machine_Synesthesia/classifierDir/'
#baseFolder = '/Users/panagiotismichalatos/Desktop/classifiers/'
imageFolder = baseFolder + 'images/train_classes/'
testFolder = baseFolder + 'images/test/'

csvFile = 'combinations.csv'


#end of parameters
#set csv file in th ecurrent python file folder
csvFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), csvFile)

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
classifierModel = createClassifierModel(classifierModel, len(classNames), freeze_pretrained_parameters = freeze_pretrained_parameters, use_pretrained = False)
classifierModel.load_state_dict(torch.load(modelSaveFile))
classifierModel.to(device)
classifierModel.eval()


data_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

#apply the model to the test images and save the results in a csv file with the image name, the predicted class and the probability of the prediction
testFiles = os.listdir(testFolder)
#remove the .DS_Store file if it exists
if '.DS_Store' in testFiles:
    testFiles.remove('.DS_Store')

testFiles = [os.path.join(testFolder, x) for x in testFiles]

with open(csvFile, 'w') as f:
    f.write('image')
    for className in classNames:
        f.write(f', {className} activation')

    for className in classNames:
        f.write(f', {className}%')
    
    f.write('\n')

    for testFile in testFiles:
        f.write(f'{testFile}')
        print(f'processing {testFile}')

        image = Image.open(testFile).convert('RGB')
        image = data_transforms(image)
        image = image.unsqueeze(0)
        image = image.to(device)
        with torch.no_grad():
            output = classifierModel(image)
            _, preds = torch.max(output, 1)
            preds = preds.cpu().numpy()
            output = output.cpu().numpy()[0]

            for i in range(len(classNames)):
                f.write(f', {output[i]}')

            output = (np.exp(output) / np.sum(np.exp(output)))*100       

            for i in range(len(classNames)):
                f.write(f', {output[i]}')
            
        f.write('\n')

# plot the results
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(csvFile)
df = df.set_index('image')
df = df.sort_index()

df = df.drop(columns = [x for x in df.columns if '%' in x])

df.plot.bar(figsize=(20,10))

# show the plot
plt.show()