import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset

from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor,Resize,RandomRotation,ColorJitter
import torch

import torchvision.transforms.functional as TF
from pathlib import Path

class PromptTrainDataset(Dataset):
    def __init__(self, args, split):
        super(PromptTrainDataset, self).__init__()
        self.args = args
        self.degrad_img_ids = []
        self.split = split
        self.detype = args.detype
        self._init_rain_ids()
        self._init_noise_ids()
        self._init_haze_ids()
       
    def _init_haze_ids(self):
        temp_ids = []
        if 'dehaze' in self.args.detype:
            if self.split == "train":
                haze_path = '../data_dir/' + "train/dehazy.txt"
                with open(haze_path, 'r') as file:
                    temp_ids += [line.strip() for line in file]   
                    
                self.haze_ids = [{'id':x, 'type':0} for x in temp_ids] 
                self.haze_ids = self.haze_ids
                self.degrad_img_ids += self.haze_ids 
                print("Total train_Haze Ids : {}".format(len(self.haze_ids)))
                
            else:
                haze_path = '../data_dir/' + "val/dehazy.txt"
                with open(haze_path, 'r') as file:
                    temp_ids += [line.strip() for line in file]
                    
                self.haze_ids = [{'id':x, 'type':0} for x in temp_ids]
                self.degrad_img_ids += self.haze_ids 
                print("Total val_Haze Ids : {}".format(len(self.haze_ids)))
                         
    def _init_rain_ids(self):
        if 'derainL' in self.args.detype:
            temp_ids = []
            if self.split == "train":
                rain_path_L = '../data_dir/' + "train/derain_L.txt"
                with open(rain_path_L, 'r') as file_L:
                    temp_ids += [line.strip() for line in file_L]   
                        
                self.rainL_ids = [{'id':x, 'type':1} for x in temp_ids]
                self.rainL_ids = self.rainL_ids * 8
                self.degrad_img_ids += self.rainL_ids
                print("Total train_rainL Ids : {}".format(len(self.rainL_ids)))
            else:
                rain_path_L = '../data_dir/' + "val/derain_L.txt"
                with open(rain_path_L, 'r') as file_L:
                    temp_ids += [line.strip() for line in file_L]   

                self.rainL_ids = [{'id':x, 'type':1} for x in temp_ids]
                self.degrad_img_ids += self.rainL_ids
                print("Total val_rainL Ids : {}".format(len(self.rainL_ids)))
        
        if 'derainH' in self.args.detype:  
            temp_ids = []
            if self.split == "train":
                rain_path_H = '../data_dir/' + "train/derain_H.txt"
                with open(rain_path_H, 'r') as file_H:
                    temp_ids += [line.strip() for line in file_H]  
                    
                self.rainH_ids = [{'id':x, 'type':2} for x in temp_ids]
                self.rainH_ids = self.rainH_ids * 8
                self.degrad_img_ids += self.rainH_ids
                print("Total train_rainH Ids : {}".format(len(self.rainH_ids)))
            
            else:
                rain_path_H = '../data_dir/' + "val/derain_H.txt"
                with open(rain_path_H, 'r') as file_H:
                    temp_ids += [line.strip() for line in file_H] 
                    
                self.rainH_ids = [{'id':x, 'type':2} for x in temp_ids]
                self.degrad_img_ids += self.rainH_ids
                print("Total val_rainH Ids : {}".format(len(self.rainH_ids)))
        
    
    
    def _init_noise_ids_bytype(self, ids):
        
        if 'denoise15' in self.args.detype:
            self.noise15_ids = [{'id':x, 'type':3} for x in ids]
            if self.split == "train":
                self.noise15_ids = self.noise15_ids * 2
            self.degrad_img_ids += self.noise15_ids
            print("Total train_noise15 Ids : {}".format(len(self.noise15_ids)))
                    
        if 'denoise25' in self.args.detype:
            self.noise25_ids = [{'id':x, 'type':4} for x in ids]
            if self.split == "train":
                self.noise25_ids = self.noise25_ids * 2
            self.degrad_img_ids += self.noise25_ids
            print("Total train_noise25 Ids : {}".format(len(self.noise25_ids)))
            
        if 'denoise50' in self.args.detype:
            self.noise50_ids = [{'id':x, 'type':5} for x in ids]
            if self.split == "train":
                self.noise50_ids = self.noise50_ids * 2
            self.degrad_img_ids += self.noise50_ids
            print("Total train_noise50 Ids : {}".format(len(self.noise50_ids)))

    def _init_noise_ids(self):
        temp_ids = []
            
        if self.split == "train":
            noise_path = '../data_dir/' + "train/denoise.txt"
            with open(noise_path, 'r') as file:
                temp_ids += [line.strip() for line in file]   
            self._init_noise_ids_bytype(temp_ids)
                
        else:
            noise_path = '../data_dir/' + "val/denoise.txt"
            with open(noise_path, 'r') as file:
                temp_ids += [line.strip() for line in file]
            self._init_noise_ids_bytype(temp_ids)
                
        
    def _add_gaussian_noise(self, image, type):
        image_array = np.array(image, dtype=np.float32)
        if type == 3:
            noise = np.random.normal(0, 15, image_array.shape).astype(np.float32)
        if type == 4:
            noise = np.random.normal(0, 25, image_array.shape).astype(np.float32)
        if type == 5:
            noise = np.random.normal(0, 50, image_array.shape).astype(np.float32) 
            
        noisy_image = image_array + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_image)

    def _random_augmentation(self, degrad_img, clean_img, scale_size, label):

        if degrad_img.size[0] > 256 and degrad_img.size[1] > 256:
            i, j, h, w = RandomCrop.get_params(degrad_img, output_size=scale_size)
            degrad_img= TF.crop(degrad_img, i, j, h, w)
            clean_img = TF.crop(clean_img, i, j, h, w)
        else:
            degrad_img = degrad_img.resize(scale_size)
            clean_img = clean_img.resize(scale_size)
        
        return degrad_img, clean_img
    
    
    def _generate_pairs(self, sample):
        id = sample['id']
        detype = sample['type']
   
        ####  dehaze:0 derainL:1 derainH:2 denoise15:3 denoise25:4 denoise50:5
        if detype in [3, 4, 5]:
            if not id or id == '.': 
                print(f"Error: Invalid ID encountered at index : '{id}'")
                raise ValueError(f"Invalid ID encountered at index : '{id}'")
            clean_img_name = id.split('.')[0].split('/')[-1]
            clean_img_path = os.path.join(self.args.data_file_dir, id)
            clean_img = Image.open(clean_img_path).convert("RGB")
            degrad_img = self._add_gaussian_noise(clean_img, detype)
            
        else :
            clean_img_name = id.split('_')[0].split('/')[-1]
            clean_img_path_base = os.path.join(self.args.data_file_dir, id.split('/')[0], id.split('/')[1] + '/clear', clean_img_name)
            clean_img_path = f"{clean_img_path_base}.jpg"
            if not os.path.exists(clean_img_path):
                clean_img_path = f"{clean_img_path_base}.png"
                
            degrad_img_path = os.path.join(self.args.data_file_dir, id)
            clean_img = Image.open(clean_img_path).convert("RGB")
            degrad_img = Image.open(degrad_img_path).convert("RGB")
            
            
        if degrad_img is None or clean_img is None:
            raise ValueError("Degrad_img or clean_img is not assigned. Please check detype value.")
        scale_size = (self.args.patch_size, self.args.patch_size) 
        if self.split != "test":
            degrad_img, clean_img = self._random_augmentation(degrad_img, clean_img, scale_size, detype)
        degrad_img = ToTensor()(degrad_img)
        clean_img = ToTensor()(clean_img)
        
        return clean_img_name, degrad_img, clean_img, detype
    
    def __getitem__(self, idx):
        sample = self.degrad_img_ids[idx]

        return self._generate_pairs(sample)

    def __len__(self):
        return len(self.degrad_img_ids)
    
class Compress_dataset(Dataset):
    def __init__(self, root, image_name, transform=None, train=True) -> None:
        super().__init__()
        self.root_dir = root
        self.transform = transform
        # self.image_names = self._get_valid_images(self.root_dir)
        self.image_names = Path(image_name).read_text().splitlines()
        self.train = train
    
    
    def __getitem__(self, idx):
        
        image_name = self.image_names[idx]
        image_path = os.path.join(self.root_dir, image_name + '.jpg')
        image = Image.open(image_path).convert("RGB")
        transform1 = Compose([RandomCrop((256, 256)), ToTensor()])
        transform2 = Compose([Resize((256, 256)), ToTensor()])
        image_width, image_height = image.size
        if self.train:
            if image_width >= 256 and image_height >= 256:
                image = transform1(image)
            else:
                image = transform2(image)
        else:
            image = ToTensor()(image)
            
        # 检查张量是否存在异常值
        if torch.isnan(image).any() or torch.isinf(image).any():
            print(f"Image '{image_name}' contains invalid values (NaN or Inf).")
        return image_name, image
      
    
    def __len__(self):
        return len(self.image_names)
    
    def _get_valid_images(self, root_dir):
        valid_images = []
        for f in os.listdir(root_dir):
            if not f.endswith(".jpg"):
                continue

            image_path = os.path.join(root_dir, f)
            try:
                with Image.open(image_path) as img:
                    if img.size[0] < 256 or img.size[1] < 256:
                        continue
                    img.verify()
                valid_images.append(f.split('.')[0])
            except(IOError, OSError):
                print(f"Warning: Skipping corrupted image: {image_path}")
                    
        return valid_images
    
        