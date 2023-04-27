import os
from PIL import Image

image_size = 256

srcFolder = '/Users/mfchao/Desktop/MachineAesthetics/classifierDir/images/src/03_10pm'
dstFolder = '/Users/mfchao/Desktop/MachineAesthetics/classifierDir/images/train_classes/03_10pm'

#ensure that the destination folder exists
if not os.path.exists(dstFolder):
    os.makedirs(dstFolder)



for filename in os.listdir(srcFolder):
    if filename.endswith('.DS_Store'):
        continue
    
    img = Image.open(os.path.join(srcFolder, filename))
    
    #resize image preserving aspect ratio
    width, height = img.size
    if width > height:
        new_width = image_size
        new_height = int((image_size / width) * height)
    else:
        new_height = image_size
        new_width = int((image_size / height) * width)

    img = img.resize((new_width, new_height))
    img.save(os.path.join(dstFolder, filename))