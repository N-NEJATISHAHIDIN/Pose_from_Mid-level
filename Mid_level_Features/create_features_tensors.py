from PIL import Image
import torchvision.transforms.functional as TF
import visualpriors
import subprocess
from torchvision.io import read_image
import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from os import walk

out_dir = "/home/negar/Documents/Datasets/pix3d/Pix3D/new_features/"
pix3d_file = "/home/negar/Documents/Datasets/pix3d/"

def make_square(im, min_size=128, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max( min_size,x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

def resize_padding(im, desired_size):
    # compute the new size
    old_size = im.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = im.resize(new_size)

    # create a new image and paste the resized on it
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))
    return new_im
    
# Choose what features you want to use, uncomment the next line 

# features = ['edge_texture','normal','autoencoding','depth_euclidean','jigsaw' ,'reshading','colorization',
# 'edge_occlusion','keypoints2d','room_layout',
# 'curvature'  ,'keypoints3d'  ,'segment_unsup2d'  ,
# 'class_object' ,'egomotion' ,  'nonfixated_pose'   , 'segment_unsup25d',
# 'class_scene',  'fixated_pose'  , 'segment_semantic',      
# 'denoising' , 'inpainting'   ,'point_matching' ,   'vanishing_point'
# ]

features = ['normal','reshading']

csv_file = open(pix3d_dir+"Pix3D/Pix3D.txt")
data_f = pd.read_csv(csv_file)

all_categories = set(data_f["cat_id"])


for feat in features:
    os.mkdir(out_dir+feat)
    for i in all_categories :
            os.mkdir(out_dir+ feat + "/"+ i)


for feat in tqdm(features):
    for ID in data_f["image_path"]:
        image = Image.open(pix3d_file+'/img_'+ID.split(".")[0]+".png")
        
        if image.mode != "RGB":
            image = Image.convert(mode='RGB')


        x = TF.to_tensor(TF.resize(image, 256)) * 2 - 1
        x = (x.unsqueeze_(0)).cuda()

        midlevel_feats = (visualpriors.representation_transform(x, feat, device='cuda:0'))[0]
        
        torch.save(midlevel_feats, out_dir+feat+ID[4:].split(".")[0]+".pt")