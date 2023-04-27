import os
from PIL import Image

image_size = 224

srcFolder = 'D:/ML/Autoencoder/images/n01443537/'
dstFolder = 'D:/ML/Autoencoder/images/train_classes/goldfish'

#ensure that the destination folder exists
if not os.path.exists(dstFolder):
    os.makedirs(dstFolder)

#load each image in the srcFolder and resize it to 128x128 and save it in the dstFolder
for filename in os.listdir(srcFolder):
    if filename.endswith('.DS_Store'):
        continue
    
    img = Image.open(os.path.join(srcFolder, filename))
    #center crop the image
    width, height = img.size
    if width > height:
        left = (width - height) / 2
        top = 0
        right = (width + height) / 2
        bottom = height
    else:
        left = 0
        top = (height - width) / 2
        right = width
        bottom = (height + width) / 2
    img = img.crop((left, top, right, bottom))
    img = img.resize((image_size,image_size))
    img.save(os.path.join(dstFolder, filename))