# Imports here
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--save_dir', action='store',
                    dest='datadirectory',
                    help='Store data_directory value')
            
parser.add_argument('--arch', action='store',
                    dest='arch',
                    help='Store Neural Network Architecture value')
                    
parser.add_argument('--learning_rate', action='store',
                    dest='lr',
                    help='Store lr (learning rate) value')
                    
parser.add_argument('--hidden_units', action='store',
                    dest='hidden_units',
                    help='Store hidden units value')

parser.add_argument('--epochs', action='store',
                    dest='epochs',
                    help='Store epochs value')
                    
parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='Set GPU Enabled to true')            

results = parser.parse_args()

print('data_directory     = {!r}'.format(results.datadirectory))
print('architecture     = {!r}'.format(results.arch))
print('learning_rate     = {!r}'.format(results.lr))
print('hidden_units     = {!r}'.format(results.hidden_units))
print('epochs     = {!r}'.format(results.epochs))
print('gpu        = {!r}'.format(results.gpu))


import helper
import numpy as np
#import matplotlib.pyplot as plt
import os, random
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import time

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
#image_datasets = datasets.ImageFolder(data_dir, transform = data_transforms)
train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
#dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size = 32, shuffle = True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = 64, shuffle = False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = False)

# Looping through it, get a batch on each loop 
for images, labels in train_loader:
    pass

# Get one batch
images, labels = next(iter(train_loader))

##### Run this to test your data loader
#images, labels = next(iter(train_loader))
#helper.imshow(images[0], normalize=True)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
#Load pre-trained network VGG
import torchvision.models as models
model = models.vgg16(pretrained=True)

#Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
#Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# mentor suggested setup
from torch import nn, optim
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(25088, 1024)
        self.fc2 = nn.Linear(1024, 102)
        
        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        # Now with dropout
        x = self.dropout(F.relu(self.fc1(x)))    

        # output so no dropout here
        x = F.log_softmax(self.fc2(x), dim=1)
        
        return x
    
# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.vgg16(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
model.classifier = Classifier()

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
#optimizer = optim.Adam(model.classifier.parameters(), lr=0.0003)

model.to(device);

images, labels = next(iter(train_loader))

epochs = 9
#epochs = 1
steps = 0
running_loss = 0
print_every = 60
for e in range(epochs):
    for images, labels in train_loader:
        steps += 1
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Move input and label tensors to the default device
        logps = model.forward(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(device), labels.to(device)
                    logps = model.forward(images)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {e+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(valid_loader):.3f}.. "
                  f"Test accuracy: {accuracy/len(valid_loader):.3f}")
            running_loss = 0
            #model.train()
            
epochs = 3
#epochs = 1
steps = 0
print_every = 60
for e in range(epochs):
    for images, labels in train_loader:
        steps += 1
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    logps = model.forward(images)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {e+1}/{epochs}.. "
                  f"Test loss: {test_loss/len(test_loader):.3f}.. "
                  f"Test accuracy: {accuracy/len(test_loader):.3f}")
            model.train()
            
model.class_to_idx = train_data.class_to_idx

print("Our model: \n\n", model, '\n')
print("The state dict keys: \n\n", model.state_dict().keys())

torch.save(model.state_dict(), 'checkpoint.pth')

# TODO: Save the checkpoint 
checkpoint = {'input_size': 25088,
              'output_size': 102,
              'arch': 'vgg16',
              'learning_rate': 0.001,
              'batch_size': 64,
              'classifier' : Classifier(),
              'epochs': 9,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_index': model.class_to_idx}

torch.save(checkpoint, 'checkpoint.pth')

model.classifier

