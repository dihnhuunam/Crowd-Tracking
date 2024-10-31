import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
from utils.density_map import gen_density_map_gaussian

class ShanghaiTechDataset(Dataset):
    def __init__(self, root_path, gt_downsample=4, is_train=True):
        self.root_path = root_path
        self.gt_downsample = gt_downsample
        self.is_train = is_train
        
        # Get all image paths
        self.image_paths = []
        self.gt_paths = []
        
        img_dir = os.path.join(root_path, 'images')
        gt_dir = os.path.join(root_path, 'ground-truth')
        
        for img_name in os.listdir(img_dir):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(img_dir, img_name)
                gt_path = os.path.join(gt_dir, f'GT_{img_name.replace(".jpg", ".mat")}')
                
                if os.path.exists(gt_path):
                    self.image_paths.append(img_path)
                    self.gt_paths.append(gt_path)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load ground truth
        gt_path = self.gt_paths[idx]
        mat = loadmat(gt_path)
        gt_coords = mat['image_info'][0, 0][0, 0][0]
        
        # Generate density map
        density_map = gen_density_map_gaussian(image, gt_coords)
        
        # Resize image and density map
        target_height = int(image.shape[0] // self.gt_downsample) * self.gt_downsample
        target_width = int(image.shape[1] // self.gt_downsample) * self.gt_downsample
        
        image = cv2.resize(image, (target_width, target_height))
        density_map = cv2.resize(density_map, 
                               (target_width // self.gt_downsample, 
                                target_height // self.gt_downsample))
        
        # Adjust density map based on resize
        density_map = density_map * ((image.shape[0] * image.shape[1]) / 
                                   (density_map.shape[0] * density_map.shape[1] * self.gt_downsample * self.gt_downsample))
        
        # Convert to tensor
        image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        density_map = torch.from_numpy(density_map).float().unsqueeze(0)
        gt_count = torch.tensor(len(gt_coords)).float()
        
        return image, density_map, gt_count