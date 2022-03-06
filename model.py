import numpy as np
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
import sys
import torch
from torch.autograd import Variable
from torch.nn import Sequential
from torch import nn
import torch.nn.functional as F



class PoseEstimationModel_baseline_NoMask_maskAzChannel_MaskOut(torch.nn.Module):
  def __init__(self, in_channels , num_bins_az , mask_size , num_bins_el, flag = None ):
    super().__init__()

    self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2)

    self.Rel = nn.ReLU()
    self.flat = nn.Flatten()
    self.fc1 = nn.Linear(32 * 7 * 7, 512)
    self.fc2 = nn.Linear(512, 128)
    self.fc3 = nn.Linear(128, num_bins_az)
    self.fc4 = nn.Linear(128, num_bins_el)
    self.flag = flag

  def forward(self, x, mask):

    if (self.flag == 1):
        x = torch.cat((x, mask),dim = 1)

    elif(self.flag == 2):
        x = mask.T*x.T
        x = x.T

    x = self.conv1(x)
    x = self.Rel(x)
    x = self.flat(x)
    x = self.fc1(x)
    x = self.fc2(x)
    x_az = self.fc3(x)
    x_el = self.fc4(x)

    return x_az, x_el


class PoseEstimationModel_baseline_NoMask_maskAzChannel_MaskOut_new(torch.nn.Module):
  def __init__(self, in_channels , num_bins_az , mask_size , num_bins_el, flag = None ):
    super().__init__()

    self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

    self.Rel = nn.ReLU()
    self.flat = nn.Flatten()
    self.fc1 = nn.Linear(64 * 63 * 63, 512)
    self.fc2 = nn.Linear(512, 128)
    self.fc3 = nn.Linear(128, num_bins_az)
    self.fc4 = nn.Linear(128, num_bins_el)
    self.flag = flag

  def forward(self, x, mask):

    if (self.flag == 1):
        x = torch.cat((x, mask),dim = 1)

    elif(self.flag == 2):
        x = mask.T*x.T
        x = x.T

    x = self.conv1(x)
    x = self.Rel(x)
    x = self.conv2(x)
    x = self.Rel(x)
    x = self.flat(x)
    x = self.fc1(x)
    x = self.fc2(x)
    x_az = self.fc3(x)
    x_el = self.fc4(x)

    return x_az, x_el



class PoseEstimationModelUpsampel_V1_NoMask_MaskedFeatures(torch.nn.Module):
  def __init__(self, in_channels,num_bins_az, mask_size,num_bins_el, flag = None):

    super().__init__()
    self.convT1 = nn.ConvTranspose2d(in_channels, 16, kernel_size=2, stride=2)
    self.convT2 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=4)
    self.convT3 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)

    self.conv1 = nn.Conv2d(8, 16, kernel_size=3, stride=2)
    self.Rel = nn.ReLU()
    self.flat = nn.Flatten()
    self.fc1 = nn.Linear(16 *  (int(mask_size/2)-1) * (int(mask_size/2)-1), 512)
    self.fc2 = nn.Linear(512, 128)
    self.fc3 = nn.Linear(128, num_bins_az)
    self.fc4 = nn.Linear(128, num_bins_el)
    self.drop = nn.Dropout(p=0.5)
    self.drop2 = nn.Dropout(p=0.2)
    self.flag = flag

  def forward(self, x, mask):

    
    x = self.convT1(x)
    x = self.convT2(x)
    # x = self.convT3(x)
    # x = self.convT3(x)
    # x = self.Rel(x)
    if (self.flag == 1):
        x = mask.T*x.T
        x = x.T
    x = self.conv1(x)
    x = self.Rel(x)
    x = self.flat(x)
    x = self.fc1(x)
    x = self.Rel(x)
    x = self.fc2(x)
    x = self.drop(x)

    x_az = self.fc3(x)
    x_el = self.fc4(x)

    return x_az, x_el


from torchvision.models import vgg,resnet50

class ResNet_NoMask(torch.nn.Module):
  def __init__(self, in_channels,num_bins_az, mask_size,num_bins_el, flag = None):

    super().__init__()
    self.convT1 = nn.ConvTranspose2d(in_channels, 16, kernel_size=2, stride=2)
    self.convT2 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=4)
    self.convT3 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)

    self.model = resnet50(pretrained=True)
    #print(self.model)
    # self.model.features = nn.Sequential(*list(self.model.features.children())[:-1])
    self.model.fc = nn.Identity()
    self.model.avgpool = nn.Identity()

    # self.vgg_seq = nn.Sequential(
    #                 nn.ConvTranspose2d(512, 16, kernel_size=4, stride=2),
    #             )   


    self.vgg_seq = nn.Sequential(
                    nn.Conv1d(512, 16, kernel_size=1, stride=1),
                )   

    self.conv1 = nn.Conv2d(8, 16, kernel_size=3, stride=2)
    self.Rel = nn.ReLU()
    self.flat = nn.Flatten()
    self.fc1 = nn.Linear(16 *  (int(mask_size/2)-1) * (int(mask_size/2)-1), 512)
    self.fc2 = nn.Linear(512, 128)
    self.fc3 = nn.Linear(128, num_bins_az)
    self.fc4 = nn.Linear(128, num_bins_el)
    self.drop = nn.Dropout(p=0.5)
    self.drop2 = nn.Dropout(p=0.2)
    self.flag = flag

  def forward(self, x, mask):

    x = self.model(x)
    #print(x.shape)
    x = x.view(-1, 512 , 256)
    #print(x.shape)
    x = self.vgg_seq(x)
    x = x.view(-1, 16 , 16,16)
    #print(x.shape)      
    x = self.convT1(x)
    x = self.convT2(x)
    # x = self.convT3(x)
    # x = self.convT3(x)
    #x = self.Rel(x)
    if (self.flag == 1):
        x = mask.T*x.T
        x = x.T
    x = self.conv1(x)
    x = self.Rel(x)
    x = self.flat(x)
    x = self.fc1(x)
    x = self.Rel(x)
    x = self.fc2(x)
    x = self.drop(x)

    x_az = self.fc3(x)
    x_el = self.fc4(x)

    return x_az, x_el


from torchvision.models import vgg

class Arslan_Paper_VGG(torch.nn.Module):
  def __init__(self, in_channels,num_bins_az, mask_size,num_bins_el, flag = None):

    super().__init__()

    self.model = vgg.vgg19_bn(pretrained=True)
    self.model.classifier = nn.Identity()

    self.az = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, num_bins_az)
                )

    self.el = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, num_bins_el)
                )

  def forward(self, x, mask):
    x = self.model(x) # 512 x 7 x 7
    x = x.view(-1, 512 * 7 * 7)
    x_az = self.az(x)
    x_el = self.el(x)

    return x_az, x_el

class Vector_Network(torch.nn.Module):
  def __init__(self, in_channels,num_bins_az, mask_size,num_bins_el, flag = None):

    super().__init__()
    self.convT1 = nn.ConvTranspose2d(in_channels, 16, kernel_size=2, stride=2)
    self.convT2 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=4)
    
    # self.convT1 = nn.Upsample(scale_factor=2, mode='nearest')
    # self.convT2 = nn.Upsample(scale_factor=4, mode='nearest')

    self.pool = nn.MaxPool2d(4)
    self.cosin   = torch.nn.CosineSimilarity()
    self.flat = nn.Flatten()


  def forward(self, x, mask, other_mask=None):
    x = self.convT1(x)
    F = self.convT2(x)
    
    x1 = self.pool(self.pool((mask.T*F.T).T))
    x2 = self.pool(self.pool((other_mask.T*F.T).T))
    return self.cosin(x1.flatten(start_dim=1),x2.flatten( start_dim=1))

class PoseEstimationModelUpsampel_V1_MaskAsChannel_pooya(torch.nn.Module):
  def __init__(self, in_channels,num_bins_az, mask_size, num_bins_el, flag = None):

    super().__init__()
    self.convT1 = nn.ConvTranspose2d(in_channels, 16, kernel_size=2, stride=2)
    self.convT2 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=4)
    self.convT3 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)
    self.bn = nn.BatchNorm1d(512)

    self.conv1 = nn.Conv2d(9, 16, kernel_size=3, stride=2)
    self.conv2 = nn.Conv2d(9, 16, kernel_size=3, stride=2)

    self.Rel = nn.ReLU()
    self.flat = nn.Flatten()

    self.fc11 = nn.Linear(32 * 63 * 63, 512)
    # self.fc22 = nn.Linear(16 * 63 * 63, 512)
    # self.fc2 = nn.Linear(1024, 512)
    self.fc3 = nn.Linear(512, 128)
    self.fc4 = nn.Linear(128, 2)
    self.fc5 = nn.Linear(128, num_bins_el)
    self.flag = flag


  def forward(self, x, mask,other_mask=None):

    x = self.convT1(x)
    x = self.convT2(x)
    # x = self.convT3(x)

    # x = self.Rel(x)

    # if(other_mask != None):
    #     x = other_mask.T*x.T
    #     x = x.T
    
    x1 = torch.cat((x,mask),dim = 1)
    x2 = torch.cat((x,other_mask),dim = 1)

    x1 = self.conv1(x1)
    x2 = self.conv2(x2)

    x = torch.cat((x1,x2),dim = 1)
    # print(x.shape)
    x = self.fc11(self.flat(self.Rel(x)))
    # x2 = self.fc22(self.flat(self.Rel(x2)))

    # x = torch.cat((x1,x2),dim = 1)
    # x = self.fc3(x)
    # x = self.bn(x)
    x = self.fc3(x)    
    x = self.Rel(x)

    # x = self.fc4(x)


    x_az = self.fc4(x)
    x_el = self.fc5(x)


    return x_az, x_el

class PoseEstimationModelUpsampel_V1_MaskAsChannel(torch.nn.Module):
  def __init__(self, in_channels,num_bins_az, mask_size, num_bins_el, flag = None):

    super().__init__()
    # self.convT1 = nn.ConvTranspose2d(16, 12, kernel_size=2, stride=2)
    # self.convT2 = nn.ConvTranspose2d(12, 8, kernel_size=4, stride=4)

    self.convT1 = nn.ConvTranspose2d(in_channels, 16, kernel_size=2, stride=2)
    self.convT2 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=4)
    # self.convT3 = nn.ConvTranspose2d(12, 8, kernel_size=2, stride=2)

    # self.convT3 = nn.Upsample(scale_factor= 8 , mode='bilinear', align_corners=True)
    self.bn = nn.BatchNorm1d(512)

    self.conv1 = nn.Conv2d(9, 16, kernel_size=3, stride=2)
    self.Rel = nn.ReLU()
    self.flat = nn.Flatten()
    self.fc1 = nn.Linear(16 * 63 * 63, 512)
    self.fc2 = nn.Linear(512, 128)

    self.cls_head = nn.Sequential(
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(16 * 63 * 63, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.Sigmoid(),
                nn.Dropout(p=0.5),
                nn.Linear(128, 1),
            )
    self.fc3 = nn.Linear(128, 1)
    self.flag = flag


  def forward(self, x, mask,gt_mask = None):

    x = self.convT1(x)
    x = self.convT2(x)
    # x = self.convT3(x)


    x = self.Rel(x)

    x = gt_mask.T*x.T
    x = x.T


    x = torch.cat((x,mask),dim = 1)

    x = self.conv1(x)
    x = self.cls_head(x)
    # print(x.shape)
    # x = self.Rel(x)
    # x = self.flat(x)
    # x = self.fc1(x)
    # # x = self.bn(x)
    # x = self.Rel(x)
    # x = self.fc2(x)

    # # x_az_diff = F.sigmoid(self.fc3(x))
    # x_az_diff = self.fc3(x)

    return x



class PoseEstimationModelUpsampel_V1_MaskAsChannel_resnet(torch.nn.Module):
  def __init__(self, in_channels,num_bins_az, mask_size, num_bins_el, flag = None):

    super().__init__()
    self.convT1 = nn.ConvTranspose2d(16, 12, kernel_size=2, stride=2)
    self.convT2 = nn.ConvTranspose2d(12, 8, kernel_size=4, stride=4)
    self.convT3 = nn.Upsample(scale_factor= 8 , mode='bilinear', align_corners=True)
    self.bn = nn.BatchNorm1d(512)

    self.convf1 = nn.Conv2d(8, 10, kernel_size=3, stride=2)
    self.convf2 = nn.Conv2d(10, 12, kernel_size=3, stride=2)
    self.ffc1 = nn.Linear(12 * 31 * 31, 2048)

    self.conv_mask1 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    self.conv_mask2 = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    self.conv1 = nn.Conv2d(9, 16, kernel_size=3, stride=2)
    self.Rel = nn.ReLU()
    self.flat = nn.Flatten()
    self.fc1 = nn.Linear(16 * 63 * 63, 512)
    self.fc2 = nn.Linear(512, 128)

    self.cls_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(4096, 1024),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 64),
                nn.BatchNorm1d(64),
                nn.Sigmoid(),
                nn.Dropout(p=0.5),
                nn.Linear(64, 1),
            )
    self.fc3 = nn.Linear(128, 1)
    self.flag = flag

    self.resnet = models.resnet50(pretrained=True)     
    self.resnet.fc = nn.Identity()



  def forward(self, x, mask,gt_mask = None):

    x = self.convT1(x)
    x = self.convT2(x)
    # x = self.convT3(x)
    # print(x.shape)

    x = self.Rel(x)

    x = gt_mask.T*x.T
    x = x.T

    x = self.convf1(x)
    x = self.convf2(x)

    x = self.flat(self.Rel(x))

    x = self.ffc1(x)

    # print(mask.shape)
    rgb_batch = mask.repeat(1,3,1,1)
    # print(rgb_batch.shape)

    mask = self.resnet(rgb_batch)
    # print(mask.shape,x.shape)
    # mask = self.conv_mask1(mask)
    # mask = self.conv_mask2(mask)
    # print(mask.shape)

    x = torch.cat((x,mask),dim = 1)
    # print(x.shape)

    # x = self.conv1(x)
    x = self.cls_head(x)
    # print(x.shape)
    # x = self.Rel(x)
    # x = self.flat(x)
    # x = self.fc1(x)
    # # x = self.bn(x)
    # x = self.Rel(x)
    # x = self.fc2(x)

    # # x_az_diff = F.sigmoid(self.fc3(x))
    # x_az_diff = self.fc3(x)

    return x


class PoseEstimationModelUpsampel_V1_MaskAsChannel_with_img(torch.nn.Module):
  def __init__(self, in_channels,num_bins_az, mask_size, num_bins_el, flag = None):

    super().__init__()
    self.convT1 = nn.ConvTranspose2d(16, 12, kernel_size=2, stride=2)
    self.convT2 = nn.ConvTranspose2d(12, 8, kernel_size=4, stride=4)
    self.convT3 = nn.Upsample(scale_factor= 8 , mode='bilinear', align_corners=True)
    self.bn = nn.BatchNorm1d(512)



    self.convimg_1 = nn.Conv2d(3, 9 , kernel_size=2, stride=2)
    # self.convimg_2 = nn.Conv2d(9, 12 ,kernel_size=3 ,padding=1, stride=1)

    self.bn1 = nn.BatchNorm2d(9)
    self.bn2 = nn.BatchNorm2d(12)

    self.conv1 = nn.Conv2d(18, 23, kernel_size=3, stride=2)    
    # self.conv2 = nn.Conv2d(23, 28, kernel_size=3, stride=2)

    self.Rel = nn.ReLU()
    self.flat = nn.Flatten()
    self.fc1 = nn.Linear(23 * 63 * 63, 512)
    self.fc2 = nn.Linear(512, 128)

    self.cls_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(23 * 63 * 63, 1024),
                nn.BatchNorm1d(512),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.Linear(512, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Sigmoid(),
                nn.Dropout(p=0.5),
                nn.Linear(64, 1),
            )
    self.fc3 = nn.Linear(128, 1)
    self.flag = flag


  def forward(self, x, mask,gt_mask,img):

    # print(img.shape)
    img = F.relu(self.bn1(self.convimg_1(img)))
    # img = F.relu(self.bn2(self.convimg_2(img)))
    # print(img.shape)

    x = self.convT1(x)
    x = self.convT2(x)
    # x = self.convT3(x)
    # print(x.shape)

    x = self.Rel(x)

    x = gt_mask.T*x.T
    x = x.T


    x = torch.cat((x,mask,img),dim = 1)

    x = self.conv1(x)
    x = self.cls_head(x)
    # print(x.shape)
    # x = self.Rel(x)
    # x = self.flat(x)
    # x = self.fc1(x)
    # # x = self.bn(x)
    # x = self.Rel(x)
    # x = self.fc2(x)

    # # x_az_diff = F.sigmoid(self.fc3(x))
    # x_az_diff = self.fc3(x)

    return x
class D_mask_selection_model(torch.nn.Module):
  def __init__(self, in_channels,num_bins_az, mask_size, num_bins_el, flag = None):

    super().__init__()
    self.convT1 = nn.ConvTranspose2d(in_channels, 16, kernel_size=2, stride=2)
    self.convT2 = nn.ConvTranspose2d(16, 12, kernel_size=2, stride=2)
    self.convT3 = nn.ConvTranspose2d(12, 8, kernel_size=2, stride=2)

    self.conv1 = nn.Conv2d(9, 16, kernel_size=3, stride=2)
    self.conv2 = nn.Conv2d(9, 16, kernel_size=3, stride=2)
    self.conv3 = nn.Conv2d(9, 16, kernel_size=3, stride=2)
    self.conv4 = nn.Conv2d(9, 16, kernel_size=3, stride=2)

    self.Rel = nn.ReLU()
    self.flat = nn.Flatten()
    self.fc1 = nn.Linear(16 * 63 * 63 * 4, 512)
    self.fc2 = nn.Linear(512, 128)
    self.fc3 = nn.Linear(128, 4)

    # self.fc4 = nn.Linear(128, 2)

    self.flag = flag


  def forward(self, x, mask1, mask2, mask3, mask4):

    x = self.convT1(x)
    x = self.convT2(x)
    x = self.convT3(x)

    # x = self.convT3(x)

    x = self.Rel(x)
    # print("x.shape",x.T.shape)
    # print("mask.shape",mask.T.shape)

    xM1 = torch.cat((x,mask1),dim = 1)
    xM2 = torch.cat((x,mask2),dim = 1)
    xM3 = torch.cat((x,mask3),dim = 1)
    xM4 = torch.cat((x,mask4),dim = 1)

    xM1 = self.conv1(xM1)
    xM2 = self.conv1(xM2)
    xM3 = self.conv1(xM3)
    xM4 = self.conv1(xM4)

    # print(x.shape)
    x = self.Rel(torch.cat((xM1,xM2,xM3,xM4)))
    x = self.flat(x)
    x = self.fc1(x)
    # x = self.bn(x)
    x = self.Rel(x)
    x = self.fc2(x)

    x_az = self.fc3(x)
    # x_el = self.fc4(x)

    return x_az, x_el


class boundries_net(torch.nn.Module):
  def __init__(self, in_channels,num_bins_az, mask_size, num_bins_el = None, flag = None):

    super().__init__()
    self.convT1 = nn.ConvTranspose2d(in_channels, 16, kernel_size=2, stride=2)
    self.convT2 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
    self.convT3 = nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2)

    self.conv1 = nn.Conv2d(5, 8, kernel_size=3, stride=2)
    self.conv2 = nn.Conv2d(5, 8, kernel_size=3, stride=2)
    self.conv3 = nn.Conv2d(5, 8, kernel_size=3, stride=2)
    self.conv4 = nn.Conv2d(5, 8, kernel_size=3, stride=2)
    

    self.Rel = nn.ReLU()
    self.flat = nn.Flatten()

    # self.fc1 = nn.Linear(32 * 31 * 31, 512)
    # self.fc2 = nn.Linear(512, 128)
    # self.fc3 = nn.Linear(128, 4)

    # # self.fc4 = nn.Linear(128, 2)

    self.fc1_1 = nn.Linear(8 * 63 * 63, 256)
    self.fc2_1 = nn.Linear(256, 64)
    self.fc3_1 = nn.Linear(64, 8)

    self.fc1_2 = nn.Linear(8 * 63 * 63, 256)
    self.fc2_2 = nn.Linear(256, 64)
    self.fc3_2 = nn.Linear(64, 8)

    self.fc1_3 = nn.Linear(8 * 63 * 63, 256)
    self.fc2_3 = nn.Linear(256, 64)
    self.fc3_3 = nn.Linear(64, 8)

    self.fc1_4 = nn.Linear(8 * 63 * 63, 256)
    self.fc2_4 = nn.Linear(256, 64)
    self.fc3_4 = nn.Linear(64, 8)


    self.convall = nn.Conv2d(32, 16, kernel_size=3, stride=2)
    self.cc = nn.Conv1d(3, 1, kernel_size=1)

    self.fc_all = nn.Linear(24, num_bins_az)



    self.flag = flag
    self.sfm = nn.Softmax(dim = 0)


  def forward(self, x, mask1, mask2, mask3, gt_mask):
    # ,  mask4

    size = x.shape[0]
    x = self.convT1(x)
    x = self.convT2(x)
    x = self.convT3(x)

    # x = self.convT3(x)

    x = self.Rel(x)
    
    x = gt_mask.T*x.T
    x = x.T
    # print(x.shape)
    # x = torch.cat((x,gt_mask),dim = 1)


    xM1 = torch.cat((x,mask1),dim = 1)
    xM2 = torch.cat((x,mask2),dim = 1)
    xM3 = torch.cat((x,mask3),dim = 1)
    # xM4 = torch.cat((x,mask4),dim = 1)

    xM1 = self.conv1(xM1)
    xM2 = self.conv2(xM2)
    xM3 = self.conv3(xM3)
    # xM4 = self.conv4(xM4)
    # print(xM4.shape)
    # # print(x.shape)
    # x = self.Rel(torch.cat((xM1,xM2,xM3,xM4),dim=1))
    # x = self.convall(x)
    # x = self.flat(x)
    # x = self.fc1(x)
    # # x = self.bn(x)
    # x = self.Rel(x)
    # x = self.fc2(x)

    # x_az = self.fc3(x)
    # # x_el = self.fc4(x)

    ###
    xM1 = self.flat(self.Rel(xM1))
    xM2 = self.flat(self.Rel(xM2))
    xM3 = self.flat(self.Rel(xM3))
    # xM4 = self.flat(self.Rel(xM4))

    xM1 = self.fc1_1(xM1)
    # x = self.bn(x)
    xM1 = self.Rel(xM1)
    xM1 = self.fc2_1(xM1)
    xM1 = self.fc3_1(xM1)
    # .reshape(size,1,8)

    xM2 = self.fc1_2(xM2)
    # x = self.bn(x)
    xM2 = self.Rel(xM2)
    xM2 = self.fc2_2(xM2)
    xM2 = self.fc3_2(xM2)
    # .reshape(size,1,8)

    xM3 = self.fc1_3(xM3)
    # x = self.bn(x)
    xM3 = self.Rel(xM3)
    xM3 = self.fc2_3(xM3)
    xM3 = self.fc3_3(xM3)
    # .reshape(size,1,8)

    # xM4 = self.fc1_4(xM4)
    # # # x = self.bn(x)
    # xM4 = self.Rel(xM4)
    # xM4 = self.fc2_4(xM4)
    # xM4 = self.fc3_4(xM4)
    # # .reshape(size,1,8)


    x = torch.cat((xM1,xM2,xM3),dim=1)
    # ,xM4
    x_az = self.fc_all(x)
    # x_az = self.flat(self.cc(x))

    return x_az





class ConvNetFeatures(torch.nn.Module):
  def __init__(self, in_channels =16,num_bins_az=9, mask_size=128, num_bins_el=5, flag = None):

    super().__init__()
    self.convT1 = nn.ConvTranspose2d(in_channels, 16, kernel_size=2, stride=2)
    self.convT2 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=4)
    self.bn = nn.BatchNorm1d(512)

    self.conv1 = nn.Conv2d(9, 12, kernel_size=3, stride=2)
    self.conv2 = nn.Conv2d(12, 15, kernel_size=3, stride=2)
    self.conv3 = nn.Conv2d(15, 20, kernel_size=3, stride=2)

    self.Rel = nn.ReLU()
    self.flat = nn.Flatten()



  def forward(self, x, mask):

    # if mask_input_bool:
    x = self.convT1(x)
    x = self.convT2(x)
    # x = self.convT3(x)
    
    x = torch.cat((x,mask),dim = 1)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.Rel(x)
    x = self.conv3(x)
    x = self.flat(self.Rel(x))
    return x

class SiameseCNN(torch.nn.Module):

    def __init__(self):
        super(SiameseCNN, self).__init__()
        self.net = ConvNetFeatures()



    def forward(self, x, mask1, mask2, mask3=None, real_mask= None):

        # features = self.net(x,None,real_mask , True)
        # anchor = self.net(x,real_mask , fa)
        output4 = None

        output1 = self.net(x,real_mask)
        output2 = self.net(x,mask1)
        output3 = self.net(x,mask2)
        if mask3 != None :
            output4 = self.net(x,mask3)

        # combined_features = output1 * output2
        # output = self.cls_head(combined_features)

        # combined_features = feat1 * feat2
        # output = self.cls_head(combined_features)

        # output2 = self.net(features,mask3)
        # output4 = self.net(features,mask4)

        return output1,output2,output3,output4


class MLP_head(torch.nn.Module):
  def __init__(self):

    super().__init__()
    self.cls_head = nn.Sequential(
                nn.Linear(14400, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 64),
                nn.BatchNorm1d(64),
                nn.Sigmoid(),
                nn.Dropout(p=0.5),
                nn.Linear(64, 1),
            )

  def forward(self, x):
    x = self.cls_head(x)
    return x
