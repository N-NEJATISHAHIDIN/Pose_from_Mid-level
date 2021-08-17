import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
import visualpriors
import subprocess
from PIL import Image, ImageOps


class PoseDataset(torch.utils.data.Dataset):
    
    def __init__(self, path, list_ids, labels, mask_size, added_feature=None, gt_D_mask_info = None):
        
        self.labels = labels
        self.list_ids = list_ids
        self.path = path
        self.mask_size = mask_size
        self.added_feature = added_feature
        self.gt_D_mask_info = gt_D_mask_info
        
    def __len__(self):
        
        return len(self.list_ids)
    
    def __getitem__(self, index):

        ID = self.list_ids[index]

        # X = Image.open(self.path  + "/Pix3D/crop/" + ID[4:].split(".")[0]+".png")
        Z = read_image(self.path  + "/Pix3D/crop_mask/" + ID[4:].split(".")[0]+".png")
        # ,interpolation = Image.NEAREST
        mask = transforms.Resize((self.mask_size,self.mask_size))(Z)
        out = mask[0].reshape(1,self.mask_size,self.mask_size)
        # print(out.shape)
        edge_temp = torch.load(self.path  + '/Pix3D/featurs/normal/'+ID[4:].split(".")[0]+'.pt', map_location=torch.device('cpu'))
        normal_temp = torch.load(self.path  + '/Pix3D/featurs/edge_texture/'+ID[4:].split(".")[0]+'.pt', map_location=torch.device('cpu'))
        features_output = torch.cat((normal_temp.float(), edge_temp.float()))
        if (self.added_feature is not None):
            add_feature = torch.load(self.path  + '/Pix3D/featurs/'+ self.added_feature +'/'+ID[4:].split(".")[0]+'.pt', map_location=torch.device('cpu'))
            features_output = torch.cat((normal_temp.float(), edge_temp.float(), add_feature.float()))
        #labels
        y =  torch.tensor((self.labels[self.labels.index.str.contains( "crop/"+ID[4:].split(".")[0])]).values[-3:])
        
        if (self.gt_D_mask_info is not None):
            im_info = self.gt_D_mask_info[self.gt_D_mask_info.index.str.contains( "crop/"+ID[4:].split(".")[0])]
            root_path = self.path + "/Pix3D/D_mask_64/"+ im_info.iloc[0][1]+ "/"+ im_info.iloc[0][2]+ "/"+ im_info.iloc[0][3]+".obj"
            full_path = root_path + "/azimuth_{}_Elevation{}.png".format(y[0][0],y[0][1])
            D_mask =  torch.from_numpy(np.asarray(Image.open(full_path).convert('1') , dtype=np.uint8)).reshape(1,64,64)
            
            #print(out.shape)
        # Z_D_mask = read_image(self.path  + "/Pix3D/D_mask/" + ID[4:].split(".")[0]+".png")
        # gt_D_mask = transforms.Resize((self.mask_size,self.mask_size))(Z)
        # out = mask[0].reshape(1,self.mask_size,self.mask_size)

        #edge_temp.float(), 
        return (features_output,D_mask.float()), (y[0][0],y[0][1],y[0][2]), ID.split(".")[0].split("/")[1],ID
