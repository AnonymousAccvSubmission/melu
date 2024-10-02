import argparse
import os
from PIL import Image
import numpy as np
import logging 
import torch
from torchvision import transforms
from .data import define_dataloader
from .models import get_generator


        
class Cleft_Inpainting_model():
    def __init__(self, pretrained_path):
        config = {'name': 'AGGV3', 'config': {'embed': 4}}
        device = torch.device('cpu')
        self.generator  = get_generator(config)
        self.generator.load_state_dict(torch.load(pretrained_path, map_location=torch.device('cpu')))
        self.generator.to(device)
        self.generator.eval()

        self.tfs = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.tfs_p = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    



    # Define your denormalize function
    def denormalize1(self, tensor):
        

        # Define the mean and std
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        # Apply denormalization: reverse normalization by multiplying with std and adding mean
        tensor = tensor * std + mean
        # Scale the pixel values from [0, 1] to [0, 255] and clamp them to valid range
        tensor = tensor * 255
        tensor = tensor.clamp(0, 255)
        tensor = tensor.squeeze(0)
        tensor = tensor.permute(1, 2, 0).byte().cpu().numpy()
        return tensor    

    def denormalize(self, tensors):
        std = np.array([0.5, 0.5, 0.5]) if tensors.shape[1] == 3 else np.array([0.5])
        mean = np.array([1, 1, 1]) if tensors.shape[1] == 3 else np.array([1])
        """ Denormalizes image tensors using mean and std """
        channels = tensors.shape[1]  # Handle both single and multi-channel cases
        for c in range(channels):
            tensors[:, c].mul_(std[c]).add_(mean[c])
        if tensors.shape[0] == 1:
            tensors = tensors.squeeze(0)  # Shape becomes (3, 256, 256)
        tensors = torch.clamp(tensors, 0, 255)
        # Convert to numpy and scale to [0, 255]
        ndarr = tensors.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    
        # Convert to a PIL image (RGB mode)
        im = np.array(Image.fromarray(ndarr, mode="RGB"))
        return im

    def predict(self, ori_img, mask_img, edge_img):
        with torch.no_grad():
            mask_img = (mask_img/255).astype(np.uint8)
            mask_img = np.reshape(mask_img, (1, 256, 256))
            ori_img = self.tfs(ori_img)
            edge_img = self.tfs_p(edge_img)
            mask_tensor = torch.from_numpy(mask_img).float()  # Convert numpy array to tensor and change dtype to float if needed

            ori_img  = ori_img.unsqueeze(0).to('cpu')
            edge_img  = edge_img.unsqueeze(0).to('cpu')
            mask_tensor  = mask_tensor.unsqueeze(0).to('cpu')
            out = self.generator(ori_img, edge_img, mask_tensor)
            out = self.denormalize1(out)
            #out = self.denormalize(out)
            return out
