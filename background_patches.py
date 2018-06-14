#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 14:32:09 2018

@author: ayooshmac
"""

import os 
import cv2 
import numpy as np
import pickle as pkl
import random

#get the list of files for training on the classification task

#Function to compute the IoU
def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    
    

    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    mask1 = (b2_y1 > b1_y2)
    mask2 = (b2_x1 > b1_x2)
    mask3 = (b1_y1 > b2_y2)
    mask4 = (b1_x1 > b2_x2)
    
    
    
    mask = 1 - (mask1*mask2*mask3*mask4)
    

    mask = mask.astype(int)

    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  np.maximum(b1_x1, b2_x1)
    inter_rect_y1 =  np.maximum(b1_y1, b2_y1)
    inter_rect_x2 =  np.minimum(b1_x2, b2_x2)
    inter_rect_y2 =  np.minimum(b1_y2, b2_y2)
    
    
    #Intersection area
    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1 + 1, 0) * np.maximum(inter_rect_y2 - inter_rect_y1 + 1, 0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    
    
    
    iou = inter_area / (b1_area)
    
    
    return iou*mask




#Load the shelf Images for background 
li = os.listdir("GroceryDataset_part1/ShelfImages")
li.remove("Thumbs.db")
li.remove("bg_images")



#Extract Selected images for Background Patches
li = [x for x in li if x[-4:] == ".JPG"]

li_bg = [x.split(".")[0] for x in li if x[:6] == "C1_P01" or x[:6] == "C2_P01"]



classes = list(range(1,11))

li_prods = []

def parse_name(name):
    a = name.split(".")[-2]
    a = a.split("_")[1:]
    x,y,w,h =  a
    return [x,y,w,h]

#Get the brand images in images for bg patches
for i in classes:
    i = str(i)
    directory = "GroceryDataset_part2/BrandImagesFromShelves/" + i
    li_prods.extend(os.listdir(directory))

dict_bg_prods = {}

#Dict {img for bg patch -> [list of bounding boxes of products]}
for img in li_bg:
    li = []
    li = [x for x in li_prods if x.split(".")[0] == img]
    
    li = list(map(parse_name, li))
    
    dict_bg_prods[img + ".JPG"] =  li
    

sizes = [280, 336]


if not os.path.exists("GroceryDataset_part1/BrandImages/0"):
    os.mkdir("GroceryDataset_part1/BrandImages/0")

for img in li_bg:
    #create 15 crops 
    i = 0
    while i < 15:
        #chose a random size for the crop
        size = random.choice(sizes)
        
        image_address = "GroceryDataset_part1/ShelfImages/" + img + ".JPG"
        
        img_ = cv2.imread(image_address)
        
        h,w,_  = img_.shape
        
        x_crop = random.randint(0, w - size - 1)
        y_crop = random.randint(0, h - size - 1)
        
        
        cropped_img = img_[y_crop:y_crop+size, x_crop:x_crop + size, :]
        
        
        #check whether there's significant overlap with a positive category
        li = dict_bg_prods[img + ".JPG"]
            
        li = np.array(li).astype(int)
        
        try:
            li[:,2], li[:,3] = li[:,0] + li[:,2], li[:,1] + li[:,3]
        
        except:
            cv2.imwrite("GroceryDataset_part1/BrandImages/0/{}_{}.jpg".format(img, i), cropped_img)
            i+=1
            continue            
        
        proposed_box = np.array([x_crop, y_crop, x_crop + size - 1, y_crop + size - 1]).astype(int)
        
        proposed_box = np.reshape(proposed_box, (1,4)) 
        

        
        max_iou = np.max(bbox_iou(proposed_box, li))
        
        if max_iou == 0:
            cv2.imwrite("GroceryDataset_part1/BrandImages/0/{}_{}.jpg".format(img, i), cropped_img)
            i+=1



    

