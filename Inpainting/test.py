import argparse
import os
from PIL import Image
import numpy as np
import logging 
from tqdm import tqdm 
import torch
from models import get_generator


        
class MymodelAGG():
    def __init__(self, pretrained_path):\
        config = {'name': 'AGGV3', 'config': {'embed': 4}}
        device = torch.device('cpu')
        self.generator  = get_generator(config)
        self.generator.load_state_dict(torch.load(pretrained_path))
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


    def predict(self, ori_img, mask_img, edge_img)
        with torch.no_grad():
            mask_img = (mask_img/255).astype(np.uint8)
            mask_img = np.reshape(mask_img, (1, 256, 256))
            ori_img = self.tfs(ori_img)
            edge_img = self.tfs_p(edge_img)
            out = generator(ori_img, mask_img, edge_img)
            return out