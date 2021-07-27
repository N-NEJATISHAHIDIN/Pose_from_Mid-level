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
from torch.utils.data import Dataset, DataLoader, TensorDataset

class PoseEstimationModel(torch.nn.Module):
  def __init__(self, in_channels):
    super(PoseEstimationModel, self).__init__()


    self.cnn_layers = Sequential(
        # Defining a 2D convolution layer
        Conv2d(in_channels, 32, kernel_size=3, stride=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32 * 7 * 7, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 8),
    )

  def forward(self, x):

    y_logits = self.cnn_layers(x)
    return y_logits


