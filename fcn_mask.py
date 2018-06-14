#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 00:43:57 2018

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
import cv2#plt.imshow(a_ > 0.5, cmap = "gray")
from PIL import Image
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from skimage import measure
import imutils
from imutils import contours


mean, std = pkl.load(open("data_mean_and_std", "rb"))
model = torch.load("Saved/vgg11_cpu.pth")

#
#img = "GroceryDataset_part1/ShelfImages/C1_P04_N3_S2_1.JPG"
#img = cv2.imread(img)
#img = Image.fromarray(img[:,:,::-1])
#img = img.resize((1120,1120))
#
#
#
#img = transforms.ToTensor()(img)
#img = img.unsqueeze(0)
#
#
#
#
#a = model(img)
#a_ = a[0].detach().numpy()
#a_ = (a_ - a_.min())/(a_.max() - a_.min())
#
#assert False
#
#a_ = pkl.load(open("a.pkl", "rb"))
#
#
##
##plt.imshow(a_ > 0.5, cmap = "gray")
##
#
#
##Dataset class for 


def map_to_bbox(map_):
    assert len(map_.shape) == 2
    
    labels = measure.label(map_, neighbors = 8, background = 0)
    
    mask = np.zeros(labels.shape, dtype = np.uint8)
    
    
    for label in np.unique(labels):
        if label == 0:
            continue
    
        LabelMask = np.zeros(labels.shape, dtype = np.uint8)
        LabelMask[labels == label] = 255
        
        if np.count_nonzero(LabelMask) > 5:
            mask = cv2.add(mask, LabelMask)

    if mask.max() == 0:
        return []
            
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = contours.sort_contours(cnts)[0]
    
    
    bboxes = []
    
    for contour in cnts:
        contour = contour.squeeze()
        xmin, ymin = contour.min(0)
        xmax, ymax = contour.max(0)
        bboxes.append((xmin, ymin, xmax, ymax))
    
    return bboxes 
    



class ShelfDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dir, size = 448, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dir = dir
        self.size = size
        self.files = os.listdir(self.dir)
        self.files = [file for file in self.files if file[-4:] == ".JPG" or file[-4:] == ".jpg"
                      or file[-4:] == ".png"]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.dir,
                                self.files[idx])
        image = cv2.imread(img_name)
        image = cv2.resize(image, (self.size,self.size))
        image = np.transpose(image, (2,0,1)).astype(np.float32)

        if self.transform:
            image = self.transform(image)
        name = self.files[idx].split(".")[0]
        return name, image


dataset = ShelfDataset("GroceryDataset_part1/ShelfImages/")
ShelfLoader = DataLoader(dataset, batch_size = 2, shuffle = True, num_workers = 0)

bbox_labels = {}
with torch.no_grad():
    for name, images in ShelfLoader:
        write_avg_maps = 0
        
        main_size = dataset.size
        sizes = [main_size, int(main_size/2), int(main_size/(4))]
        for size_ in sizes:
            images = nn.Upsample(size = (size_, size_), mode = "bilinear")(images)
            
            maps = model(images)
            
            maps = nn.Upsample(size = main_size, mode = "bilinear")(maps)
            
            if not write_avg_maps:
                avg_maps = maps
                write_avg_maps = 1
            else:
                avg_maps += maps
                
            
    
        
        seg_maps = avg_maps/len(sizes)
    
        for i, map_ in enumerate(seg_maps):
            
            bbox_labels[name[i]] = []
            
            map_max = map_.max(0)[0]
            
            map_max = map_max.unsqueeze(0)
            
            map_ *= (map_ == map_max).float()
            
            for cls in range(1,11):
                
                cls_map = map_[cls].numpy()
                
                
                bounding_boxes = map_to_bbox(cls_map > 0.5)
                
                if len(bounding_boxes) > 0:
                    bbox_labels[name[i]].append((int(cls), bounding_boxes))
                    
            a = cv2.imread("GroceryDataset_Part1/ShelfImages/{}.JPG".format(name[i]))
            plt.imshow(a[:,:,::-1])
            print(bbox_labels)
            assert False
    
        
        
            
            
        
        
        
        
    





