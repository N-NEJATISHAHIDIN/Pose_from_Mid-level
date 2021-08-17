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
            root_path = self.path + "/Pix3D/D_mask/"+ im_info.iloc[0][1]+ "/"+ im_info.iloc[0][2]+ "/"+ im_info.iloc[0][3]+".obj"
            D_mask =  np.array(Image.open(root_path + "/azimuth_{}_Elevation{}.png".format(y[0][0],y[0][1])).convert('1'))
            #print((D_mask==True).any())
            points_real_mask = np.asarray(np.where(D_mask == 1))
            #print(points_real_mask)
            max_x, max_y = np.max(points_real_mask, 1)
            min_x, min_y = np.min(points_real_mask, 1)
            #print(max_x, max_y , min_x, min_y)

            dim_y = max_y - min_y
            dim_x = max_x - min_x

            dim = np.max([dim_x , dim_y])
            #print(dim)
            crop_image = torch.from_numpy(D_mask[ min_x : max_x, min_y : max_y])
            
            crop_D_mask = F.pad(crop_image, pad=((dim - dim_y)//2, (dim - dim_y)//2, (dim - dim_x)//2, (dim - dim_x)//2) )
            crop_D_mask = crop_D_mask.reshape(1, crop_D_mask.shape[0], crop_D_mask.shape[1])
            #print("source_pad.shape:" , crop_D_mask.shape)
            out = transforms.Resize((self.mask_size,self.mask_size))(crop_D_mask)
            # print(out.shape)

            #print(out.shape)
        # Z_D_mask = read_image(self.path  + "/Pix3D/D_mask/" + ID[4:].split(".")[0]+".png")
        # gt_D_mask = transforms.Resize((self.mask_size,self.mask_size))(Z)
        # out = mask[0].reshape(1,self.mask_size,self.mask_size)

        #edge_temp.float(), 
        return (features_output,out.float()), (y[0][0],y[0][1],y[0][2]), ID.split(".")[0].split("/")[1],ID
