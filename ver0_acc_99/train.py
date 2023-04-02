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

#一些基础设置

torch.multiprocessing.set_start_method('spawn')
PATH = 'C:\\Users\\scmmm\\Desktop\\mmmtorch\\AI\\0DigitalRecognizer\\'
TRAIN = 'train.csv'
TEST = 'test.csv'
MODEL = 'model.pth'
SUBMISSION = 'submissions.csv'
NUM_WORKERS = 0

# 读入数据

train_origin = pd.read_csv(PATH+TRAIN)
test_origin = pd.read_csv(PATH+TEST)

# 读入数据后，此时对train再处理
one_hot_embedding =  pd.get_dummies(train_origin.label,prefix='y')

# axis = 1 是横着合并
train_origin = pd.concat([one_hot_embedding,train_origin],axis=1)

# axis ? 索引 : 列
train_origin = train_origin.drop(['label'],axis=1)

# check the answer 
print(train_origin)

# 现在就是划分训练集和测试集
# train , valid
# 还没有了解过这个函数，不知道是不是随机划分的，但是考虑到数据集也是随机的应该没啥关系
# stratify表示是否按照数据分布划test和val

x_train, x_val, y_train, y_val = train_test_split(train_origin.iloc[:, 10:], train_origin.iloc[:, 0:10], train_size = 0.92, stratify=train_origin.iloc[:, 0:10])
x_train = x_train/255.0
x_val = x_val/255.0
train_dataset = pd.concat([x_train, y_train], axis=1).reset_index(drop=True)
val_dataset = pd.concat([x_val, y_val], axis=1).reset_index(drop=True)

# 检测
# train_sample = train_dataset.sample(5)
# val_sample = val_dataset.sample(5)
# train_features, train_label = train_sample.iloc[:, 0:784], train_sample.iloc[:, 784:]
# val_features, val_label = val_sample.iloc[:, 0:784], val_sample.iloc[:, 784:]
# fig, axs = plt.subplots(2, 5, figsize=(10, 5))

# axes = axs.ravel()

# fig.suptitle('Split check')

# train_heatmaps = [[np.array(list(row)[1]), list(train_label.iloc[idx, :]).index(max(train_label.iloc[idx, :])), 'train'] for idx, row in enumerate(train_features.iterrows())]
# val_heatmaps = [[np.array(list(row)[1]), list(val_label.iloc[idx, :]).index(max(val_label.iloc[idx, :])), 'validation'] for idx, row in enumerate(val_features.iterrows())]

# concat_list = train_heatmaps + val_heatmaps

# for i, (arr, label, dataset) in enumerate(concat_list):
#     axs = axes[i]
    
#     sns.heatmap(arr.reshape((28, 28)), ax=axs)
#     axs.set_xlabel(label)
#    axs.set_title(f'{dataset}_sample')
#    
#plt.tight_layout()
#plt.show()
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
# 我并不知道 totensorV2后还会不会共享内存，但是它真的快，虽然这模型没跑到GPU瓶颈
# 在这里，我们设置了一个图像随机旋转，0.5的概率旋转不超过25度

train_pytorch_dataset = MnistDataset(df=train_dataset, transform=A.Compose([
    A.Rotate(limit=17.5, p=0.8),
    ToTensorV2(),
]))
validate_pytorch_dataset = MnistDataset(df=val_dataset, transform=A.Compose([
    ToTensorV2(),
]))

# 打乱一下读入，感觉意义不大
#  dataset,batch_size,shuffle,num_workers,

train_dataloader = DataLoader(train_pytorch_dataset, batch_size=64, shuffle=True,num_workers=NUM_WORKERS)
val_dataloader = DataLoader(validate_pytorch_dataset, batch_size=64, shuffle=True,num_workers=NUM_WORKERS)

# hyperparaments

loss_fn = nn.CrossEntropyLoss()
epochs =  20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Simple_CNN(input_shape=(1, 28, 28), output_classes=10).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0003)

# train

summary(net,(1,28,28))

#print(net)

train_acc, train_loss, val_acc, val_loss = fit(model=net,
                                               loss_fn=loss_fn,
                                               optimizer=optimizer,
                                               epochs=epochs, 
                                               train_dataloader=train_dataloader, 
                                               val_dataloader=val_dataloader, 
                                               device=device)

# 验证数据
# compose 里面还可以加上其他数据增强的东西，比如随即仿射变换等等。

PAT = PATH+MODEL
torch.save(net.state_dict(),PAT)
net.load_state_dict(torch.load(PAT))

test_origin/=255.0

test_pytorch_dataset = MnistDataset(df=test_origin, transform=A.Compose([
    ToTensorV2(),
]))

test_dataloader = DataLoader(test_pytorch_dataset, batch_size=1, shuffle=False,num_workers=NUM_WORKERS)

def test(model, test_dataloader, device):
    
    ids = []
    preds = []
    
    model.eval()

    tmp = 0 
    target = len((test_dataloader))

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_dataloader):
            tmp += 1
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

preds_df.to_csv(PATH+SUBMISSION, index=False)

getscore1 = pd.read_csv(PATH+'answer.csv')
getscore1 = getscore1.loc[ : ,"Label"]
# if check
# getscore1 = getscore1[:999]
#
getscore2 = pd.read_csv(PATH+SUBMISSION)
getscore2 = getscore2.loc[ : ,"Label"]

tot = (getscore1==getscore2).sum()
score = tot / len(getscore2)
print(score)