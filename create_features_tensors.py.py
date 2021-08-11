from PIL import Image
import torchvision.transforms.functional as TF
import visualpriors
import subprocess
from torchvision.io import read_image
import torch
import pandas as pd
import numpy as np
import os
import tqdm


features = ['autoencoding','depth_euclidean','jigsaw' ,'reshading','colorization',
'edge_occlusion','keypoints2d','room_layout',
'curvature' ,'edge_texture' ,'keypoints3d'  ,'segment_unsup2d'  ,
'class_object' ,'egomotion' ,  'nonfixated_pose'   , 'segment_unsup25d',
'class_scene',  'fixated_pose'  ,  'normal'  , 'segment_semantic',      
'denoising' , 'inpainting'   ,'point_matching' ,   'vanishing_point'
]

csv_file = open("../../../Datasets/pix3d/Pix3D/Pix3D.txt")
data_f = pd.read_csv(csv_file)

all_categories = set(data_f["cat_id"])


for feat in features:
    os.mkdir("/home/negar/Documents/Datasets/pix3d/Pix3D/featurs/+feat")
    for i in all_categories :
            os.mkdir("/home/negar/Documents/Datasets/pix3d/Pix3D/featurs/"+ feat + "/"+ i)


for feat in tqdm(features):
    for ID in data_f["image_path"]:
        image = Image.open('../../../Datasets/pix3d/Pix3D/'+ID.split(".")[0]+".png")
        
        x = TF.to_tensor(TF.resize(image, 256)) * 2 - 1
        x = (x.unsqueeze_(0)).cuda()

        midlevel_feats = (visualpriors.representation_transform(x, 'edge_texture', device='cuda:0'))[0]
        #normal_temp = (visualpriors.representation_transform(x, 'normal', device='cuda:0'))[0]
        
        torch.save(midlevel_feats, "/home/negar/Documents/Datasets/pix3d/Pix3D/featurs/"+feat+ID[4:].split(".")[0]+".pt")
        #torch.save(normal_temp , "../../../Datasets/pix3d/Pix3D/featurs/normal"+ID[4:].split(".")[0]+".pt")

