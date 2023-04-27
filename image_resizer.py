import os
from PIL import Image

srcFolder = 'D:/ML/Autoencoder/images/n01443537/'
dstFolder = 'D:/ML/Autoencoder/images/train/'

#ensure that the destination folder exists
if not os.path.exists(dstFolder):
    os.makedirs(dstFolder)

#load each image in the srcFolder and resize it to 128x128 and save it in the dstFolder
for filename in os.listdir(srcFolder):
    img = Image.open(os.path.join(srcFolder, filename))
    img = img.resize((64,64))
    img.save(os.path.join(dstFolder, filename))