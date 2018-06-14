#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 14:54:20 2018

@author: ayooshmac
"""

import cv2
import numpy as np
import pickle as pkl
import os


def get_mean():
    """
    Get the mean of the images in the entire dataset
    """
    classes = list(map(lambda x: str(x),list(range(0,11))))
    
    data_dir = "GroceryDataset_part1/BrandImages/"
    
    classes_dir = ["{0}{1}/".format(data_dir, clas) for clas in classes]
    
    li = []
    
    for class_dir in classes_dir:
        im_path = [os.path.join(class_dir, x) for x in os.listdir(class_dir)]
        li.extend(im_path)
    
    li = [x for x in li if x[-4:] == ".jpg"]
    
    mean = np.array([0.0,0.0,0.0])
    
    i = 0
    
    large_matrix = False

    for image in li:
        
        print(i)
        i += 1
        
        if image[-2:] == "db":
            continue
        
        img = cv2.imread(image)
        img=  cv2.resize(img, (224,224))
#        mean_ = np.sum(img, 0)
#        mean_ = np.sum(mean_, 0)
#        mean_= mean_/(224*224*255)
#        mean += mean_

        img = img.reshape(-1,3)
        
        
        if not large_matrix:
            comb = img
            large_matrix = True
        
        else:
            comb = np.concatenate((comb, img), 0)
            print(comb.shape)
        
        
            

        
    mean = np.mean(comb, 0)/255
    std = np.std(comb, 0)/255
    return (mean, std)

mean, std = get_mean()

pkl.dump((mean, std), open("data_mean_and_std", "wb"))

