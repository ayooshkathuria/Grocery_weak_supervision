#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 19:53:22 2018

@author: ayooshmac
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import pickle as pkl
import copy
from torch.utils.data.sampler import SubsetRandomSampler



mean, std = pkl.load(open("data_mean_and_std", "rb"))
plt.ion()  


#


#Set up the data augmentation
#Data Augmentation is done on the fly

affine = torchvision.transforms.RandomAffine(10, translate = (0.2, 0.2), 
                                             scale = (0.8, 1), shear = 5)

hsv = torchvision.transforms.ColorJitter(brightness=0.25, contrast=0.25,hue=0.2)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(0.4),
        torchvision.transforms.RandomApply([hsv], 0.25),
        torchvision.transforms.RandomApply([affine], 0.3),
        transforms.Resize((112,112)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),

    'val': transforms.Compose([
        transforms.Resize((112,112)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}


#Load the  datasets
data_dir = "GroceryDataset_part1/BrandImages/"
train_dataset =  datasets.ImageFolder(data_dir, data_transforms['train'])
valid_dataset =  datasets.ImageFolder(data_dir, data_transforms['val'])

#Set the validation split 
valid_size = 0.2
# random_seed = 5  #for reproducibility 


num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

# np.random.seed(random_seed)
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

#Set the parameters 

batch_size = 32
num_workers= 4
pin_memory = True

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )


valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

class_names = train_dataset.classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    
inputs, classes = next(iter(train_loader))

out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])

dataloaders = {"train": train_loader, "val": valid_loader}

dataset_sizes = {"train": len(train_dataset) - split, "val": split }


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.
            running_corrects = 0

            # Iterate over data.
            i = 0
            for inputs, labels in dataloaders[phase]:                
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).squeeze()
                    _, preds = torch.max(outputs, 1)
                    
                    
                    print(outputs.shape)
                    
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                i += batch_size


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        
model_ft = models.vgg11(pretrained=True)

    
    
arch = "vgg"

if arch == "res":    
    modified_model_ft = nn.Sequential(*list(model_ft.children())[:-1])
else:
    modified_model_ft = nn.Sequential(*list(model_ft.features.children())[:-1])
    

modified_model_ft.add_module("Final",nn.Conv2d(512, 11, 7))

model_ft = modified_model_ft

model_ft = model_ft.to(device)



criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9, weight_decay = 0.0005)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#
#model_ft.load_state_dict(torch.load("Saved/Model_cpu.pth"))
#
#torch.save(model_ft, "Saved/Whole_Model.pth")

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=5)

#torch.save(model_ft, open("Model.pth", "wb"))


visualize_model(model_ft)





