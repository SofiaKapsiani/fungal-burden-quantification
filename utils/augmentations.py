# code for performing data augmentations
import cv2
import albumentations as A
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
import random

from torch.utils.data import Dataset
import torch



# define transformations
transform = A.Compose([
                # geometric augmentations
                A.Rotate(limit=180, p=0.9),
                A.HorizontalFlip(p=0.8),
                A.VerticalFlip(p=0.8),
                # colour space augmentations
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
                   A.HueSaturationValue(hue_shift_limit=(-5, 0), sat_shift_limit=5, val_shift_limit=5, p=0.5)
                ], p=0.8),
                A.Sharpen( alpha=(0.4,0.8), lightness=(0.6,0.8), p=0.5),
                ])

class AugDataset(Dataset):
    
    @classmethod
    def augment_data(cls, data, transform=transform, num=2):
        
        # initiate lists to store augmented images and masks
        image_list = []
        mask_list = []
        
        # iterate through data
        for i in data:
            image = i["image"]
            mask = i["mask"]
            
            # add original images and masks into the lists
            image_list.append(image)
            mask_list.append(mask)
           
            
            # set seed to ensure reproducibility
            random.seed(8)
            
            # apply transformations and store images/masks to respective lists
            if transform is not None:
                for i in range(num):
                    augmentations = transform(image=image, mask=mask)
                    augmented_img = augmentations["image"]
                    augmented_mask = augmentations["mask"]
                    image_list.append(augmented_img)
                    mask_list.append(augmented_mask)

        return image_list, mask_list
    
    @classmethod
    def preprocess(cls, img_nd):

        # convert gray scale image channels from two to three channels
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
            
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans
    
    def __init__(self, data, transform=transform, num=2):
        # get data
        self.data = data
        # get transformations
        self.transform = transform
        # get number of transformed images to be generated for each image in the train set
        self.num = num
        # augment images and masks and store into seperate lists
        self.images, self.masks = self.augment_data(self.data, self.transform, self.num)

    # get length of dataset (including augmented images_
    def __len__(self):
        return len(self.images)
    
    # export dictonary with images and corresponing masks in tensor form
    def __getitem__(self, index):
        
        img = self.preprocess(self.images[index])
        mask = self.preprocess(self.masks[index])
        
        return {'image': torch.from_numpy(img).type(torch.FloatTensor),
                'mask':torch.from_numpy(mask).type(torch.FloatTensor)}