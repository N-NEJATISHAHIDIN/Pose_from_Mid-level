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



def make_square(im_batch, min_size=256, fill_color=(0, 0, 0, 0)):
    new_imgs =[]
    for im in im_batch:
        x, y = im.size
        size = max(min_size, x, y)
        new_im = Image.new('RGBA', (size, size), fill_color)
        new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
        print(new_im.shape)
        new_imgs.append(new_im)
    return new_imgs

class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_ids, labels,overlap_label, mask_size,data_file,home):
        
        self.home = home
        self.labels = labels
        self.overlap_label = overlap_label
        self.list_ids = list_ids
        self.path = path
        self.mask_size = mask_size
        self.data_file = data_file
        self.im_path = "/home/negar/Documents/label_AVD_Pose/crop_padd/"

    def __len__(self):
        return len(self.list_ids)
    

    def __getitem__(self, index):

        ID = self.list_ids[index]


        ID = self.list_ids[index]
        img_rgb_path = ID.split(".")[0]
        img_rgb = read_image(self.im_path + self.home+"/"+img_rgb_path+".png")

        edge_temp = torch.load("/home/negar/Documents/Projects/ICRA_pose_on_pix3d/AVD_feat/reshading/"+self.home+img_rgb_path+'.pt', map_location=torch.device('cpu'))[0]
        normal_temp = torch.load("/home/negar/Documents/Projects/ICRA_pose_on_pix3d/AVD_feat/normal/"+  self.home+img_rgb_path+'.pt', map_location=torch.device('cpu'))[0]

        # edge_temp = torch.load("/home/negar/Documents/label_AVD_Pose/featurs/reshading/"+self.home+img_rgb_path+'.pt', map_location=torch.device('cpu'))
        # normal_temp = torch.load("/home/negar/Documents/label_AVD_Pose/featurs/normal/"+  self.home+img_rgb_path+'.pt', map_location=torch.device('cpu'))

        # print(normal_temp.shape,edge_temp.shape)
        features_output = torch.cat((normal_temp.float(), edge_temp.float()))

        y =  torch.tensor((self.labels.loc[ID])).unsqueeze(0)
        y_over =  torch.tensor((self.overlap_label.loc[ID]).values[-1:])


        return (features_output.float(), torch.tensor([])), y[0],y_over[0], ID.split("/")[0],ID,img_rgb