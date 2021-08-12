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
    
    def __init__(self, path, list_ids, labels, mask_size, added_feature):
        
        self.labels = labels
        self.list_ids = list_ids
        self.path = path
        self.mask_size = mask_size
        self.added_feature = added_feature
        
    def __len__(self):
        
        return len(self.list_ids)
    
    def __getitem__(self, index):

        ID = self.list_ids[index]

        # X = Image.open(self.path  + "/Pix3D/crop/" + ID[4:].split(".")[0]+".png")
        Z = read_image(self.path  + "/Pix3D/crop_mask/" + ID[4:].split(".")[0]+".png")
        # ,interpolation = Image.NEAREST
        mask = transforms.Resize((self.mask_size,self.mask_size))(Z)
        out = mask[0].reshape(1,self.mask_size,self.mask_size)

        edge_temp = torch.load(self.path  + '/Pix3D/featurs/normal/'+ID[4:].split(".")[0]+'.pt', map_location=torch.device('cpu'))
        normal_temp = torch.load(self.path  + '/Pix3D/featurs/edge_texture/'+ID[4:].split(".")[0]+'.pt', map_location=torch.device('cpu'))
        add_feature = torch.load(self.path  + '/Pix3D/featurs/'+ self.added_feature +'/'+ID[4:].split(".")[0]+'.pt', map_location=torch.device('cpu'))
        
        #labels
        y =  torch.tensor((self.labels[self.labels.index.str.contains( "crop/"+ID[4:].split(".")[0])]).values[-3:])
        #edge_temp.float(), 
        try:
            return (torch.cat((normal_temp.float(), edge_temp.float(), add_feature.float())),out.float()), y[0][0]
            # , out.float()
        except:
            print(y,ID)