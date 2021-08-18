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
def generate_label(data, n_bins, n_bins_elev = 5):
    
    #generate the bins margin
    _,bins_degree = np.histogram(1, bins=n_bins, range=(0,360))
    _,bins_degree_2 = np.histogram(1, bins=n_bins_elev, range=(0,100))
    _,bins_radian = np.histogram(1, bins=n_bins, range=(-math.pi,math.pi))
    

    reg_label = np.asarray(data)[:,-3:]

        
    label = data
    #reg_label[reg_label[:,0]<0] += 360 
    out = np.digitize(reg_label[:,0],bins_degree,right=True)-1
    out1 = np.digitize(reg_label[:,1],bins_degree_2,right=True)-1
    out2 = np.digitize(reg_label[:,2],bins_radian,right=True)-1

    out1[out1[:]==-1] = 0
    out2[out2[:]==-1] = 0
    
       
    #out[out[:]>7] = 0
    #out1[out1[:]>2] = 0
    #out2[out2[:]>3] = 0
    
    label["az"] =out
    label["ele"]=out1
    label["inp"]=out2
    label = label.set_index([0])
    return label[["az","ele","inp"]]
    
    
def get_Dmask(az,el,ID,gt_D_mask_info):
    masks = torch.empty(az.shape[0],64,64)
    path = "../../Datasets/pix3d"
    for i in range(az.shape[0]):
        im_info = gt_D_mask_info[gt_D_mask_info.index.str.contains( "crop/"+ID[i][4:].split(".")[0])]
        root_path = path + "/Pix3D/D_mask_64/"+ im_info.iloc[0][1]+ "/"+ im_info.iloc[0][2]+ "/"+ im_info.iloc[0][3]+".obj"
        full_path = root_path + "/azimuth_{}_Elevation{}.png".format(az[i],el[i])
        D_mask =  torch.from_numpy(np.asarray(Image.open(full_path).convert('1') , dtype=np.uint8))
        masks[i] = D_mask

    return masks.reshape(az.shape[0],1,64,64)

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
    
    