import numpy as np
import math
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import visualpriors
import subprocess
from PIL import Image, ImageOps
import torch.nn.functional as F
from torchvision import transforms
import os
from torchvision.utils import save_image
from tqdm import tqdm
import sys
import json

from pathlib import Path
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).

def rotationMatrixToEulerAngles(R) :
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])


# generate a list of labels for the dataset.[real pose_x, real pose_y, real pose_z, bin of pose_x, bin of pose_y, bin of pose_z]
def generate_label(data, n_bins, n_bins_elev ):
    
    #generate the bins margin
    _,bins_degree_org = np.histogram(1, bins=n_bins-1, range=(360/n_bins/2,360-(360/n_bins/2)))
    # (360/n_bins/2,360-(360/n_bins/2)
    overlap = (360/(n_bins-1) - 360/n_bins )/2
    # bins_degree_over = [ 20-2.5,20+2.5, 60-2.5,60+2.5, 100-2.5,100+2.5, 140-2.5,140+2.5, 180-2.5,180+2.5, 220-2.5,220+2.5, 260-2.5,260+2.5, 300-2.5, 300+2.5, 340-2.5, 340+2.5]
    bins_degree_over = []
    for i in range(bins_degree_org.shape[0]):
        bins_degree_over.append(bins_degree_org[i]-overlap)
        bins_degree_over.append(bins_degree_org[i]+overlap)
    print(bins_degree_over)
    # bins_degree_over = [ 22.5-5, 22.5+5, 67.5-5,67.5+5, 112.5-5, 112.5+5, 157.5-5,157.5+5, 202.5-5,202.5+5, 247.5-5, 247.5+5, 292.5-5,292.5+5, 337.5-5,337.5+5]

    _,bins_degree_2 = np.histogram(1, bins=n_bins_elev, range=(0,100))
    _,bins_radian = np.histogram(1, bins=n_bins, range=(-math.pi,math.pi))
    # print(bins_degree_over)

    reg_label = np.asarray(data)[:,-3:]

        
    label = data
    over_label = data

    out = np.digitize(reg_label[:,0],bins_degree_org, right=True)

    out_over = np.digitize(reg_label[:,0],bins_degree_over, right=True)

    out1 = np.digitize(reg_label[:,1],bins_degree_2,right=True)-1
    out2 = np.digitize(reg_label[:,2],bins_radian,right=True)-1

    
    out[out[:]==0] = 0
    out[out[:]==n_bins] = 0
    out1[out1[:]==-1] = 0
    out2[out2[:]==-1] = 0


    out_over[out_over[:]==0] = 0
    out_over[out_over[:]==n_bins*2] = 0
    #out[out[:]>7] = 0
    #out1[out1[:]>2] = 0
    #out2[out2[:]>3] = 0
    # label_az = np.zeros((len(out),3))

    # label_az[:,0] = out 
    # label_az[:,1] = out_over 
    # label_az[:,2] = reg_label[:,0]
    # print(label_az)
    label["az"] = out
    label["ele"]=out1
    label["inp"]=out2
    label = label.set_index([0])

    over_label["az"] = out_over
    over_label["ele"]=out1
    over_label["inp"]=out2
    over_label = over_label.set_index([0])

    return label[["az","ele","inp"]], over_label[["az","ele","inp"]]
    
def generate_label_AVD(data, n_bins ):
    
    #generate the bins margin
    _,bins_degree_org = np.histogram(1, bins=n_bins-1, range=(360/n_bins/2,360-(360/n_bins/2)))

    bins_degree_over = [ 20-2.5,20+2.5,  60-2.5,60+2.5, 100-2.5,100+2.5, 140-2.5,140+2.5, 180-2.5,180+2.5, 220-2.5,220+2.5, 260-2.5,260+2.5, 300-2.5, 300+2.5, 340-2.5, 340+2.5]
    reg_label = np.asarray(data)[:,0]

    label = data
    over_label = data

    out = np.digitize(reg_label,bins_degree_org, right=True)

    out_over = np.digitize(reg_label,bins_degree_over, right=True)

    
    out[out[:]==0] = 0
    out[out[:]==n_bins] = 0

    out_over[out_over[:]==0] = 0
    out_over[out_over[:]==18] = 0
    label["az"] = out
    label = label.set_index(['filename'])

    over_label["az"] = out_over
    over_label = over_label.set_index(['filename'])

    return label[["az"]], over_label[["az"]]
    
    
    
def get_Dmask(az,el,ID,gt_D_mask_info,mask_size = 128):
    masks = torch.empty(az.shape[0],mask_size,mask_size)
    path = "../../Datasets/pix3d"
    for i in range(az.shape[0]):
        im_info = gt_D_mask_info[gt_D_mask_info.index.str.contains( "crop/"+ID[i][4:].split(".")[0])]
        root_path = path + "/Pix3D/crop_D_mask_9_bin/"+ im_info.iloc[0][1]+ "/"+ im_info.iloc[0][2]+ "/"+ im_info.iloc[0][3]+".obj"
        full_path = root_path + "/azimuth_{}_Elevation{}.png".format(az[i],el[i])
        D_mask =  torch.from_numpy(np.asarray(Image.open(full_path).convert('1') , dtype=np.uint8))
        # print(D_mask)
        masks[i] = D_mask

    return masks.reshape(az.shape[0],1,mask_size,mask_size)

def get_Dmask_AVD(az,el,ID,mask_size = 128):
    masks = torch.empty(az.shape[0],mask_size,mask_size)
    path = "../../Datasets/pix3d"
    for i in range(az.shape[0]):
        root_path = path + "/Pix3D/crop_D_mask_9_bin/"+ID[i]
        full_path = root_path + "azimuth_{}_Elevation{}.png".format(az[i],el[i])
        D_mask =  torch.from_numpy(np.asarray(Image.open(full_path).convert('1') , dtype=np.uint8))
        # print(D_mask)
        masks[i] = D_mask

    return masks.reshape(az.shape[0],1,mask_size,mask_size)


def generate_Dmask(mask_size = 64):
    
    inputpath = "/home/negar/Documents/Datasets/pix3d/Pix3D/D_mask/"
    output_base = "/home/negar/Documents/Datasets/pix3d/Pix3D/D_mask_128/"
    
    for dirpath, dirnames, filenames in os.walk(inputpath):
        for files in tqdm(filenames):
            #print(files,dirpath.split("D_mask"))

            D_mask =  np.array(Image.open(dirpath + "/"+files).convert('1'))
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
            out = transforms.Resize((mask_size,mask_size))(crop_D_mask)    
            #print(type(out))
            #print(out[0].shape)
            save_image(out[0].float(), output_base+dirpath.split("D_mask")[1] +"/" + files )
    
    
    
def get_model_config(model_name):
    
    f = open('config.json',)
    data = json.load(f)[model_name] 

    return data

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def save_top_masks(mask_path, img_name, top, elev,prob,ID,gt_D_mask_info):
    mask_size = 128
    path = "../../Datasets/pix3d"
    im_info = gt_D_mask_info[gt_D_mask_info.index.str.contains( "crop/"+ID[4:].split(".")[0])]
    root_path = path + "/Pix3D/crop_D_mask_9_bin/"+ im_info.iloc[0][1]+ "/"+ im_info.iloc[0][2]+ "/"+ im_info.iloc[0][3]+".obj"
    print(root_path)
    inputpath = "/home/negar/Documents/Datasets/pix3d/Pix3D/D_mask/"
    output_base = "/home/negar/Documents/Datasets/pix3d/Pix3D/D_mask_top_final3/"
    Path(output_base + img_name).mkdir(parents=True, exist_ok=True)
    i = top
    # print(top)
    for (ind,i) in enumerate(top):
        img = Image.open(root_path +"/azimuth_{}_Elevation{}.png".format(str(i.item()),str(elev.item())))
        img.save(output_base + img_name + "/azimuth_{}_Elevation{}_{}.png".format(str(i.cpu().item()),str(elev.cpu().item()),str(prob[ind].cpu().item())))

    img = Image.open(mask_path)
    img.save( output_base + img_name + mask_path[-25:-4]+"_label.png" )

    img2 = Image.open("/home/negar/Documents/Datasets/pix3d/Pix3D/crop_real_masks/" + img_name+ ".png")
    img2.save( output_base + img_name + "/real.png")

    # img2 = Image.open("/home/negar/Documents/Datasets/pix3d/Pix3D/crop/" + img_name+ ".png")
    # img2.save( output_base + img_name + "/" + img_name[-4:] +".png")

   
   