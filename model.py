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


class PoseEstimationModel_baseline(torch.nn.Module):
  def __init__(self, in_channels,num_bins,num_bins_el):
    super().__init__()

    self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2)

    self.Rel = nn.ReLU()
    self.flat = nn.Flatten()
    self.fc1 = nn.Linear(32 * 7 * 7, 512)
    self.fc2 = nn.Linear(512, 128)
    self.fc3 = nn.Linear(128, num_bins)
    self.fc4 = nn.Linear(128, num_bins_el)


  def forward(self, x,mask,flag):
    #x = torch.cat(x,mask)
    x = self.conv1(x)
    x = self.Rel(x)
    x = self.flat(x)
    x = self.fc1(x)
    x = self.fc2(x)
    x_az = self.fc3(x)
    x_el = self.fc4(x)

    # x = self.Rel(x)
    return x_az, x_el

class PoseEstimationModel_MaskedFeatures(torch.nn.Module):
  def __init__(self, in_channels,num_bins):

    super().__init__()

    self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2)
    self.Rel = nn.ReLU()
    self.flat = nn.Flatten()
    self.fc1 = nn.Linear(32 * 7 * 7, 512)
    self.fc2 = nn.Linear(512, 128)
    self.fc3 = nn.Linear(128, num_bins)



  def forward(self, x, mask,flag):

    x = mask.T*x.T
    x = self.conv1(x.T)
    x = self.Rel(x)
    x = self.flat(x)
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    return x

class PoseEstimationModelUpsampel_V1_MaskedFeatures(torch.nn.Module):
  def __init__(self, in_channels,num_bins_az, mask_size,num_bins_el):

    super().__init__()
    self.convT1 = nn.ConvTranspose2d(in_channels, 12, kernel_size=2, stride=2)
    self.convT2 = nn.ConvTranspose2d(12, 8, kernel_size=2, stride=2)
    # self.convT3 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)

    self.conv1 = nn.Conv2d(8, 16, kernel_size=3, stride=2)
    self.conv1 = nn.Conv2d(8, 16, kernel_size=3, stride=2)
    self.Rel = nn.ReLU()
    self.flat = nn.Flatten()
    self.fc1 = nn.Linear(16 * 31 * 31, 512)
    self.fc2 = nn.Linear(512, 128)
    self.fc3 = nn.Linear(128, num_bins_az)
    self.fc4 = nn.Linear(128, num_bins_el)

  def forward(self, x, mask,flag):

    # print(mask_size)
    x = self.convT1(x)
    x = self.convT2(x)
    # x = self.convT3(x)
    x = self.Rel(x)
    if (flag == 1):
        x = mask.T*x.T
        x = x.T
    x = self.conv1(x)
    # print(x.shape)
    x = self.Rel(x)
    x = self.flat(x)
    x = self.fc1(x)
    x = self.Rel(x)
    x = self.fc2(x)

    x_az = self.fc3(x)
    x_el = self.fc4(x)

    # x = self.Rel(x)
    return x_az, x_el

class PoseEstimationModelUpsampel_V1_MaskAsChannel(torch.nn.Module):
  def __init__(self, in_channels,num_bins, mask_size, num_bins_el):

    super().__init__()
    self.convT1 = nn.ConvTranspose2d(in_channels, 12, kernel_size=2, stride=2)
    self.convT2 = nn.ConvTranspose2d(12, 8, kernel_size=2, stride=2)
    # self.convT3 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)
    self.bn = nn.BatchNorm1d(512)

    self.conv1 = nn.Conv2d(9, 16, kernel_size=3, stride=2)
    self.Rel = nn.ReLU()
    self.flat = nn.Flatten()
    self.fc1 = nn.Linear(16 * 31 * 31, 512)
    self.fc2 = nn.Linear(512, 128)
    self.fc3 = nn.Linear(128, num_bins)
    self.fc4 = nn.Linear(128, num_bins_el)


  def forward(self, x, mask,flag):

    x = self.convT1(x)
    x = self.convT2(x)
    # x = self.convT3(x)

    x = self.Rel(x)
    # print("x.shape",x.T.shape)
    # print("mask.shape",mask.T.shape)

    x = torch.cat((x,mask),dim = 1)
    x = self.conv1(x)
    # print(x.shape)
    x = self.Rel(x)
    x = self.flat(x)
    x = self.fc1(x)
    # x = self.bn(x)
    # x = self.Rel(x)
    x = self.fc2(x)

    x_az = self.fc3(x)
    x_el = self.fc4(x)

    return x_az, x_el


# class PoseEstimationModelUpsampel_V2_MaskedFeatures(torch.nn.Module):
#   def __init__(self, in_channels, num_bins):

#     super().__init__()

#     # self.convT1 = nn.ConvTranspose2d(in_channels, 8, kernel_size=2, stride=2)
#     # self.convT2 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)
#     # # self.convT3 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)
#     # # self.bn = nn.BatchNorm2d()
#     self.conv1 = nn.Conv2d(8, 16, kernel_size=3, stride=2)
#     # self.Rel = nn.ReLU()
#     self.flat = nn.Flatten()
#     self.fc1 = nn.Linear(16 * 31 * 31, 512)
#     self.convT1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#     self.fc2 = nn.Linear(512, 128)
#     self.fc3 = nn.Linear(128, num_bins)

#   def forward(self, x, mask, flag):

#     # print(mask_size)
#     x = self.convT1(x)
#     x = self.convT2(x)
#     # x = self.convT3(x)
#     x = self.bn(x)
#     x = self.Rel(x)
#     if (flag == 1):
#         x = mask.T/255*x.T
#         x = x.T
#     x = self.conv1(x)
#     # print(x.shape)
#     x = self.Rel(x)
#     x = self.flat(x)
#     x = self.fc1(x)
#     x = self.Rel(x)
#     x = self.fc2(x)
#     x = self.fc3(x)
#     # x = self.Rel(x)
#     return x



