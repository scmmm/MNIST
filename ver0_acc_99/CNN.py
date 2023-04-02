from typing import Tuple, Union

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import torch.nn.functional as F

from torch.utils.data import SubsetRandomSampler

import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class Simple_CNN(nn.Module):
    def __init__(self, input_shape: tuple = (1, 28, 28), output_classes: int = 10):
        super().__init__()
        
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=7,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        output_h, output_w = self._calculate_output_shape(input_shape = (input_shape[1], input_shape[2]), sequential_block=self.conv_layer_1)
                
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        output_h, output_w = self._calculate_output_shape(input_shape = (output_h, output_w), sequential_block=self.conv_layer_2)
        
        self.flatten = nn.Flatten()
        
        self.fc = nn.Sequential(
            nn.Linear(output_h * output_w * 64, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
        )
        
        self.classifier = nn.Linear(1024, output_classes)
        
    def _calculate_output_shape(self, input_shape: Tuple[int, int], sequential_block: nn.Sequential) -> Tuple[int, int]:
     
        output_shape = input_shape

        for layer in sequential_block:

            if isinstance(layer, nn.Conv2d):
                output_shape = (
                    (output_shape[0] - layer.kernel_size[0] + 2 * layer.padding[0]) // layer.stride[0] + 1,
                    (output_shape[1] - layer.kernel_size[1] + 2 * layer.padding[1]) // layer.stride[1] + 1
                )
                
            if isinstance(layer, nn.MaxPool2d):

                output_shape = (
                    (output_shape[0] - layer.kernel_size) // layer.stride + 1,
                    (output_shape[1] - layer.kernel_size) // layer.stride + 1
                )

        return output_shape
        
    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.fc1(x)
        x = self.classifier(x)
        return x
