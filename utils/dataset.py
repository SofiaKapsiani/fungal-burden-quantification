# code for adding images and mask into PyTorch tensors, adapted from GitHub repository: "https://github.com/milesial/Pytorch-UNet"
from os.path import splitext
from os import listdir
import numpy as np
import re
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        # get images directory
        self.imgs_dir = imgs_dir
        # get masks directory
        self.masks_dir = masks_dir
        # add mask suffix if necessary
        self.mask_suffix = mask_suffix
        # define scale
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        
        # get images ids, ignoring hidden files
        self.ids = [re.split('.jpg', file)[0] for file in listdir(imgs_dir)
                    if not file.startswith(('.', 'desktop'))]
        # print dataset length
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    # get dataset length
    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        # rescale input image
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
        
        # convert image to numpy array
        img_nd = np.array(pil_img)

        return img_nd

    def __getitem__(self, i):
        # iterate through images
        idx = self.ids[i]
        # obtain mask 
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        # obtain histology image
        img_file = glob(self.imgs_dir + idx + '.*')
        
        # ensure all images have corresponding masks and vice versa
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        # extract image and mask files
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])
        # ensure image and mask have the same dimentions
        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
 
        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': img,
            'mask': mask,
        }
