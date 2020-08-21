import json
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os

r'''
root = '../lab5_dataset/iclevr/'

data = json.load(open(os.path.join(root,'test.json')))

obj = json.load(open(os.path.join(root,'objects.json')))
        
img = list(data.keys())
        
label = list(data.values())
        
for i in range(len(label)):
    for j in range(len(label[i])):
        label[i][j] = obj[label[i][j]]
img_name = np.squeeze(img)
img   = Image.open(os.path.join(root,img_name[i])).convert('RGB')
trans = transforms.Compose([
            transforms.Resize([64,64]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                               std=[0.5, 0.5, 0.5])
        ])
trans(img)
'''

'''
root = '../lab5_dataset/iclevr/'

data = json.load(open(os.path.join(root,'test.json')))

obj = json.load(open(os.path.join(root,'objects.json')))
    
label = data
        
for i in range(len(label)):
    for j in range(len(label[i])):
        label[i][j] = obj[label[i][j]]
len(label)
'''


def getData(root,mode):
    if mode == 'train':
        data = json.load(open(os.path.join(root,'train.json')))
        obj = json.load(open(os.path.join(root,'objects.json')))
        img = list(data.keys())
        label = list(data.values())
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
        return np.squeeze(img), np.squeeze(label)
    else:
        data = json.load(open(os.path.join(root,'test.json')))
        obj = json.load(open(os.path.join(root,'objects.json')))
        label = data
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
        return None, label


class ICLEVRLoader(data.Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.img_name, self.label = getData(root,mode)
        self.mode = mode
        self.num_classes = 24
        if self.mode == 'train':
            print("> Found %d images..." % (len(self.img_name)))
        
        # Transform:Convert the pixel value to [0, 1]
        #           Transpose the image shape from [H, W, C] to [C, H, W]
        # We can just use transform.ToTensor() to acheive it
        self.trans = transforms.Compose([
            transforms.Resize([64,64]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                               std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        """'return the size of dataset"""
        return len(self.label)

    def __getitem__(self, index):
        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        if self.mode == 'train':
            img   = Image.open(os.path.join(self.root,self.img_name[index])).convert('RGB')
            if self.trans is not None:
                img = self.trans(img)
            label = self.label[index]
            one_hot_y = torch.zeros(self.num_classes)
            for i in label:
                one_hot_y[i] = 1.
            label = one_hot_y

            return img, label
        elif self.mode == 'test':
            label = self.label[index]
            one_hot_y = torch.zeros(self.num_classes)
            for i in label:
                one_hot_y[i] = 1.
            label = one_hot_y
            
            return label
