import cv2
import numpy as np
import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import RandomResizedCrop
import torchvision.transforms.functional as TF
from monai.transforms import Compose, ToTensor, RandFlip

def imread_kor ( filePath, mode=cv2.IMREAD_UNCHANGED ) : 
    stream = open( filePath.encode("utf-8") , "rb") 
    bytes = bytearray(stream.read()) 
    numpyArray = np.asarray(bytes, dtype=np.uint8)
    return cv2.imdecode(numpyArray , mode)
    
def imwrite_kor(filename, img, params=None): 
    try: 
        ext = os.path.splitext(filename)[1] 
        result, n = cv2.imencode(ext, img, params) 
        if result:
            with open(filename, mode='w+b') as f: 
                n.tofile(f) 
                return True
        else: 
            return False 
    except Exception as e: 
        print(e) 
        return False
        
class ImagesDataset(Dataset):
    def __init__(self, image_path_list, target_path_list, aug=False):
        self.image_path_list = image_path_list
        self.target_path_list = target_path_list
        self.transform = transforms.Compose([
                        transforms.ToTensor(),
                ])
        self.aug = aug
    def __len__(self):
        return len(self.image_path_list)
 
    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        mask_path = self.target_path_list[idx]
        image = imread_kor(image_path)[:,:,:3]
        image = self.transform(image).float()
        
        mask = imread_kor(mask_path)
        mask = torch.tensor(mask.sum(2)>0).float().unsqueeze(0)
        
        if self.aug==True:
            if random.random() < 0.4:
                resize_transform = RandomResizedCrop(size=(512, 512))
                i, j, h, w = resize_transform.get_params(image, scale=(0.3, 1.0), ratio=(1, 1))
                image = TF.resized_crop(image, i, j, h, w, (512, 512))
                mask = TF.resized_crop(mask, i, j, h, w, (512, 512))
            if np.random.randint(2)==1:
                image = RandFlip(1, 0)(image)
                mask = RandFlip(1, 0)(mask)
        mask[mask > 0] = 1
        return image, mask, image_path