import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision import transforms
import numpy as np

from PIL import Image
import torchvision.transforms.functional as TF
import visualpriors
import subprocess

class PoseDataset(torch.utils.data.Dataset):
    
    def __init__(self, path, list_ids, labels):
        
        self.labels = labels
        self.list_ids = list_ids
        self.path = path

        
    def __len__(self):
        
        return len(self.list_ids)
    
    def __getitem__(self, index):

        ID = self.list_ids[index][5]

        if(ID[-4:] == "jpeg" or ID[-4:] == "tiff"):
            
            # read RGB, read mask, read boundries, read normals
            X = read_image(self.path  + "/Pix3D/crop/"+ID[5:-4]+"png")
            Z = read_image(self.path  + "/Pix3D/crop_mask/"+ID[5:-4]+"png")[1,:,:]
            
          
        else:
            
            # read RGB, read mask, read boundries, read normals
            X = read_image(self.path  + "/Pix3D/crop/"+ID[5:-4]+"png")
            Z = read_image(self.path  + "/Pix3D/crop_mask/"+ID[5:-3]+"png")[1,:,:]
       
        
        x = TF.to_tensor(TF.resize(X, 256)) * 2 - 1
        x = (x.unsqueeze_(0)).cuda()

        # Transform to normals feature
        edge_temp = (visualpriors.representation_transform(x, 'edge_texture', device='cuda:0'))[0]
        normal_temp = (visualpriors.representation_transform(x, 'normal', device='cuda:0'))[0]

        #labels   
        y =  torch.tensor((self.labels.loc["crop/"+ID[5:]]).values[-3:])
        return (edge_temp, normal_temp), y      
    
    
    
    
    
    
    
    
    
