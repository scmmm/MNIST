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

def fit(model, loss_fn, optimizer, epochs, train_dataloader, val_dataloader, device):
    
    PAT = 'C:\\Users\\scmmm\\Desktop\\mmmtorch\\AI\\0DigitalRecognizer\\model.pth'
    torch.save(model.state_dict(),PAT)
    model.train()
    
    loss_fn = loss_fn.cuda()
    
    train_acc, train_loss, val_acc, val_loss = [], [], [], []
    
    lastbest=0.0
    
    for epoch in range(epochs):
        
        train_loss_batch, train_acc_batch = 0, 0
        val_loss_batch, val_acc_batch = 0, 0
        tmp=0
        looker=0
        for inputs, labels in train_dataloader:
            
            tmp=tmp+1

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            inputs = inputs.to(torch.float32)
            labels = labels.to(torch.float32)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            output = model(inputs)

            loss = loss_fn(output, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss_batch += loss.item()
            

            # 返回最大值和索引
            _, predicted = torch.max(output, 1)

            _, y_true = torch.max(labels, 1)

            correct = (predicted == y_true).sum().item()
            
            train_acc_batch += correct / len(labels)

            looker=train_acc_batch/tmp
            

            print('\r [{:03d}/{:03d}] {:03d} Train Acc: {:3.6f}  '.format(
                epoch+1 , epochs ,tmp ,looker
            ),end='')            
        train_average_accuracy = train_acc_batch / len(train_dataloader)
        train_average_loss = train_loss_batch / len(train_dataloader)
                
        train_loss.append(train_average_loss)
        train_acc.append(train_average_accuracy)
        
        model.eval()
        
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                inputs = inputs.to(torch.float32)
                labels = labels.to(torch.float32)
                
                output = model(inputs)
                loss = loss_fn(output, labels)
                
                val_loss_batch += loss.item()
                
                _, predicted = torch.max(output, 1)
            
                _, y_true = torch.max(labels, 1)
                
                correct = (predicted == y_true).sum().item()
            
                val_acc_batch += correct / len(labels)
                
            val_average_accuracy = val_acc_batch / len(val_dataloader)
            val_average_loss = val_loss_batch / len(val_dataloader)
                
            val_loss.append(val_average_loss)
            val_acc.append(val_average_accuracy)

            print('\n{:3.6f} :{:3.6f}\n ',val_average_accuracy,lastbest)
            if val_average_accuracy<lastbest :
                model.load_state_dict(torch.load(PAT))
                print('[{:03d}/{:03d}] |Acc: {:3.6f} (unchanged)'.format(
                epoch+1 , epochs , lastbest
                ), end='\n')
            else :
                torch.save(model.state_dict(),PAT)
                lastbest = val_average_accuracy
                print('[{:03d}/{:03d}] | train Acc: {:3.6f} loss: {:3.6f} | test Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch+1 , epochs , train_average_accuracy,  train_average_loss,val_average_accuracy,val_average_loss,
                ), end='\n')   
                    
    return train_acc, train_loss, val_acc, val_loss