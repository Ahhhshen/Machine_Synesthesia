import torch
import torch.optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import autoencoder
import numpy as np
import os
from PIL import Image

#set the parameters for the model training
image_size = 64
image_channels = 3
modelname = "Simple_01"

baseFolder = '/Users/panagiotismichalatos/Desktop/Autoencoder/'
#baseFolder = 'D:/ML/Autoencoder/'

imageFolder = baseFolder + 'images/train/'
modelSaveFolder = baseFolder + f'models/{modelname}/'
modelSaveFile = os.path.join(modelSaveFolder, 'model_weights.pth')

#select the model to train
autoencoderModel  = autoencoder.SimpleAutoencoder((image_size,image_size), inputChannels=image_channels)
#autoencoderModel = autoencoder.VGG16Autoencoder((image_size,image_size), latentSize=400, inputChannels=image_channels)
#autoencoderModel = autoencoder.VariationalAutoencoder((image_size,image_size), latentSize=120, inputChannels=image_channels)

num_epochs = 20

#check if cuda is available for acceleration of the training process
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#initialize the autoencoder
autoencoderModel.to(device)
autoencoderModel.initWeights()
autoencoderModel.train()

#initialize the tensorboard logging system
writer = SummaryWriter(modelSaveFolder)

#log the model architecture
print(autoencoderModel)
autoencoderModel.printShapes((image_size, image_size), device)

#initialize the dataloader
transform = transforms.Compose([
    transforms.Resize((image_size,image_size)), 
    transforms.ToTensor(),
    autoencoder.normXForm
    ])
dataset = autoencoder.ImageDataset(imageFolder, image_channels=image_channels, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)


#initialize the optimizer
optimizer = torch.optim.Adam(autoencoderModel.parameters(), lr=0.001, weight_decay=1e-5)

#train the model
for epoch in range(num_epochs):
    for img in dataloader:
        #________________________load data (get the image from the dataloader)
        img = img.to(device)       
        #________________________zero the parameter gradients (reset the gradients)
        optimizer.zero_grad()         
        #________________________forward (pass the image through the model)
        output = autoencoderModel(img)
        #________________________compute loss (compare the output to the original image)
        loss= autoencoderModel.computeLoss(output, img)
        #________________________backward (compute the gradients)
        loss.backward()
        #________________________optimize (update the weights)
        optimizer.step()

    #________________________log the loss at each epoch for monitoring
    writer.add_scalar('Loss/train', loss, epoch)
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))

writer.flush()
#save the model
if not os.path.exists(modelSaveFolder):
    os.makedirs(modelSaveFolder)
torch.save(autoencoderModel.state_dict(), modelSaveFile)

#model visualization
autoencoderModel.eval()

#initialize the embedding projector
image_count = min(100, len(dataset))
embedding = torch.zeros(image_count, autoencoderModel.latentSize)
embedding_images = torch.zeros(image_count, 3, image_size, image_size)

#select a random sample of images from the dataset
subset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), image_count, replace=False))

#loop through the dataset and get the latent space for each image
for i, img in enumerate(subset):
    img = img.unsqueeze(0).to(device)    
    latent = autoencoderModel.encode(img).flatten()
    embedding[i] = latent
    embedding_images[i] = img


#save the embedding projector
writer.add_embedding(embedding, label_img=embedding_images, global_step=0)

writer.add_graph(autoencoderModel, img)

writer.close()
