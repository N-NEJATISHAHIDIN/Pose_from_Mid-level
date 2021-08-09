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

# class PoseEstimationModelMaskedFeatures(torch.nn.Module):
#   def __init__(self, in_channels,num_bins):

#     super().__init__()

#     self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2)
#     self.Rel = nn.ReLU()
#     self.flat = nn.Flatten()
#     self.fc1 = nn.Linear(32 * 7 * 7, 512)
#     self.fc2 = nn.Linear(512, 128)
#     self.fc3 = nn.Linear(128, num_bins)



#   def forward(self, x):

#     x = x[:,-1,:,:].T/255*x[:,:-1,:,:].T
#     x = self.conv1(x.T)
#     x = self.Rel(x)
#     x = self.flat(x)
#     x = self.fc1(x)
#     x = self.fc2(x)
#     x = self.fc3(x)
#     return x



class PoseEstimationModel(torch.nn.Module):
  def __init__(self, in_channels,num_bins):
    super(PoseEstimationModel, self).__init__()

    self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2)

    self.Rel = nn.ReLU()
    self.flat = nn.Flatten()
    self.fc1 = nn.Linear(32 * 7 * 7, 512)
    self.fc2 = nn.Linear(512, 128)
    self.fc3 = nn.Linear(128, num_bins)


  def forward(self, x):

    x = self.conv1(x)
    x = self.Rel(x)
    x = self.flat(x)
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)

    return x

