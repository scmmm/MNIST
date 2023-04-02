from typing import Tuple, Union

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv
import random
from fit import fit
from CNN import Simple_CNN

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

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from PIL import Image
from torchsummary import summary

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# 读入数据

test_origin = pd.read_csv('C:\\Users\\scmmm\\Desktop\\mmmtorch\\AI\\0DigitalRecognizer\\test.csv')

# 现在就是划分训练集和测试集
# train , valid
# 还没有了解过这个函数，不知道是不是随机划分的，但是考虑到数据集也是随机的应该没啥关系
# stratify表示是否按照数据分布划test和val

# 然后就是搞一个数据集

class MnistDataset(Dataset):
    """
    This class is made for the mnist dataset we have currently. This is specifically made for a PyTorch DataLoader.
    
    Args:
        df (pd.DataFrame): A pandas dataframe consisting of either a training dataset or a validation dataset.
        transform (A.Compose): An Albumentations pipeline object where we define all the transformation methods we are using.
    """
    
    def __init__(self, df: pd.DataFrame, transform: A.Compose = None):
        self.df = df
        self.transform = transform
        
    def __len__(self) -> int:
        """
        This function returns the length of our dataframe object.
        
        Returns:
            The length of our dataframe.
        """
        return self.df.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        pixel_series, label = self.df.iloc[idx, :784], self.df.iloc[idx, 784:]
        
        pixel_numpy = pixel_series.to_numpy().reshape(28, 28)
        
        if self.transform:
            transformed = self.transform(image=pixel_numpy)["image"]
            
        return transformed, np.array(label)

# 加载数据
# numpy 和 torch 对于图像的表示不同，numpy是 HW C(Channel) ，而torch 是 CHW ，因此需要to tensor
# 我并不知道 totensorV2后还会不会共享内存，但是它真的快，不过这代码没泡到GPU瓶颈。
# 在这里，我们设置了一个图像随机旋转，0.5的概率旋转不超过20度

# hyperparaments

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Simple_CNN(input_shape=(1, 28, 28), output_classes=10).to(device)

# 验证数据
# compose 里面还可以加上其他数据增强的东西，比如随即仿射变换等等。

PAT = 'C:\\Users\\scmmm\\Desktop\\mmmtorch\\AI\\0DigitalRecognizer\\model.pth'
net.load_state_dict(torch.load(PAT))

test_origin/=255.0

test_pytorch_dataset = MnistDataset(df=test_origin, transform=A.Compose([
    ToTensorV2(),
]))

test_dataloader = DataLoader(test_pytorch_dataset, batch_size=1, shuffle=False)

def test(model, test_dataloader, device):
    
    ids = []
    preds = []
    
    model.eval()
    
    tmp = 0 
    target = len((test_dataloader))

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_dataloader):
            tmp +=1
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs = inputs.to(torch.float32)
            labels = labels.to(torch.float32)

            output = model(inputs)

            _, predicted = torch.max(output, 1)
            
            print('\r[{:03d}/{:03d}] '.format(
            tmp , target
            ), end='')

            ids.append(idx + 1)
            preds.append(predicted.item())
        
    return ids, preds

ids, pred = test(model=net, test_dataloader=test_dataloader, device=device)

preds_df = pd.DataFrame({"ImageId": ids, "Label": pred})



preds_df.to_csv('C:\\Users\\scmmm\\Desktop\\mmmtorch\\AI\\0DigitalRecognizer\\submissions.csv', index=False)

getscore1 = pd.read_csv('C:\\Users\\scmmm\\Desktop\\mmmtorch\\AI\\0DigitalRecognizer\\answer.csv')
getscore1 = getscore1.loc[ : ,"Label"]
# if check
# getscore1 = getscore1[:999]
#
getscore2 = pd.read_csv('C:\\Users\\scmmm\\Desktop\\mmmtorch\\AI\\0DigitalRecognizer\\submissions.csv')
getscore2 = getscore2.loc[ : ,"Label"]

print(getscore1)
print(getscore2)


tot = (getscore1==getscore2).sum()
score = tot / len(getscore2)
print(score)


