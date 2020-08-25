import time
import PIL
import os, sys
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
import torch.optim as optim

import pandas as pd

import datasets
import sys

import models


EPOCHS = 1

def train_shadow_model(dataset, lr, hidden_attribute, class_distribution, device, size=2000, filename='test'):
    dataloader = datasets.get_dataloader(dataset, hidden_attribute, size, class_distribution)
    net = models.Net10().to(device)

    tic = time.time()
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=lr)

    losses = []
    sum_0 = 0
    sum_1 = 0
    for epoch in range(EPOCHS):
        running_loss = 0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data

            total_0 = (labels == 0).sum()
            total_1 = (labels == 1).sum()
            sum_0 += total_0
            sum_1 += total_1

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs).to(device)

            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        losses.append(running_loss/(i+1))

    tac = time.time()
    print(f'[{epoch+1}] loss: {running_loss/(i+1):.3f} - {int(tac-tic)} sec')

    # path = f'./models/{filename}.pth'
    # torch.save(net.state_dict(), path)


def train_shadow_models(dataset, device, rep=1500, lr=0.001, hidden_attribute='Male'):
    print(f'\nStarting training of {rep} shadow models on {device} - lr={lr}')
    threshold = 70
    data = {'model': [], 'male_dist': [], 'split': [], 'architecture': []}
    df = pd.read_csv('./models/models.csv')
    print(len(df))
    for model in range(len(df), len(df) + rep, 2):
        # target property p is whether dataset is composed of more than 70% of male images
        p_true = np.random.randint(threshold, 101)
        p_false = np.random.randint(0, threshold)
        p_true_distribution = np.array([100-p_true, p_true])/100
        p_false_distribution = np.array([100-p_false, p_false])/100

        try:
            # Train model with property p
            print(f'{model} - p=True - d={p_true_distribution} - Training...')
            filename = f'{model}'
            train_shadow_model(dataset=dataset,
                               lr=lr,
                               hidden_attribute=hidden_attribute,
                               class_distribution=p_true_distribution,
                               device=device,
                               size=2000,
                               filename=filename)
            data['model'].append(f'{filename}.pth')
            data['male_dist'].append(p_true/100)
            data['split'].append(0)
            data['architecture'].append('a10')
        except:
            break

        try:
            # Train model without property p
            print(f'{model+1} - p=False - d={p_false_distribution} - Training...')
            filename = f'{model+1}'
            train_shadow_model(dataset=dataset,
                               lr=lr,
                               hidden_attribute=hidden_attribute,
                               class_distribution=p_false_distribution,
                               device=device,
                               size=2000,
                               filename=filename)
            data['model'].append(f'{filename}.pth')
            data['male_dist'].append(p_false/100)
            data['split'].append(0)
            data['architecture'].append('a10')
        except:
            break

    # new_df = pd.DataFrame(data).set_index(pd.Index(range(len(df), len(df)+len(data['model']))))
    # new_df.to_csv('./models/models.csv', mode='a', header=False)

def train_shadow_models_test(dataset, device, rep=200, lr=0.001, hidden_attribute='Male'):
    print(f'\nStarting training of {rep} shadow models test on {device} - lr={lr}')
    threshold = 70
    data = {'model': [], 'male_dist': [], 'split': [], 'architecture': []}
    df = pd.read_csv('./models/models.csv')
    print(len(df))
    for model in range(len(df), len(df) + rep, 2):
        # target property p is whether dataset is composed of more than 70% of male images
        p_true = np.random.randint(threshold, 101)
        p_false = np.random.randint(0, threshold)
        p_true_distribution = np.array([100-p_true, p_true])/100
        p_false_distribution = np.array([100-p_false, p_false])/100

        try:
            # Train model with property p
            print(f'{model} - p=True - d={p_true_distribution} - Training...')
            filename = f'{model}'
            train_shadow_model(dataset=dataset,
                               lr=lr,
                               hidden_attribute=hidden_attribute,
                               class_distribution=p_true_distribution,
                               device=device,
                               size=2000,
                               filename=filename)
            data['model'].append(f'{filename}.pth')
            data['male_dist'].append(p_true/100)
            data['split'].append(1)
            data['architecture'].append('a10')
        except:
            break

        try:
            # Train model without property p
            print(f'{model+1} - p=False - d={p_false_distribution} - Training...')
            filename = f'{model+1}'
            train_shadow_model(dataset=dataset,
                               lr=lr,
                               hidden_attribute=hidden_attribute,
                               class_distribution=p_false_distribution,
                               device=device,
                               size=2000,
                               filename=filename)
            data['model'].append(f'{filename}.pth')
            data['male_dist'].append(p_false/100)
            data['split'].append(1)
            data['architecture'].append('a10')
        except:
            break

    # new_df = pd.DataFrame(data).set_index(pd.Index(range(len(df), len(df)+len(data['model']))))
    # new_df.to_csv('./models/models.csv', mode='a', header=False)

def train_shadow_models_valid(dataset, device, rep=100, lr=0.001, hidden_attribute='Male'):
    print(f'\nStarting training of {rep} shadow models valid on {device} - lr={lr}')
    threshold = 70
    data = {'model': [], 'male_dist': [], 'split': [], 'architecture': []}
    df = pd.read_csv('./models/models.csv')
    print(len(df))
    for model in range(len(df), len(df) + rep, 2):
        print(model)
        # target property p is whether dataset is composed of more than 70% of male images
        p_true = np.random.randint(threshold, 101)
        p_false = np.random.randint(0, threshold)
        p_true_distribution = np.array([100-p_true, p_true])/100
        p_false_distribution = np.array([100-p_false, p_false])/100

        try:
            # Train model with property p
            print(f'{model} - p=True - d={p_true_distribution} - Training...')
            filename = f'{model}'
            train_shadow_model(dataset=dataset,
                               lr=lr,
                               hidden_attribute=hidden_attribute,
                               class_distribution=p_true_distribution,
                               device=device,
                               size=2000,
                               filename=filename)
            data['model'].append(f'{filename}.pth')
            data['male_dist'].append(p_true/100)
            data['split'].append(2)
            data['architecture'].append('a10')
        except:
            break

        try:
            # Train model without property p
            print(f'{model+1} - p=False - d={p_false_distribution} - Training...')
            filename = f'{model+1}'
            train_shadow_model(dataset=dataset,
                               lr=lr,
                               hidden_attribute=hidden_attribute,
                               class_distribution=p_false_distribution,
                               device=device,
                               size=2000,
                               filename=filename)
            data['model'].append(f'{filename}.pth')
            data['male_dist'].append(p_false/100)
            data['split'].append(2)
            data['architecture'].append('a10')
        except:
            break

    # new_df = pd.DataFrame(data).set_index(pd.Index(range(len(df), len(df)+len(data['model']))))
    # new_df.to_csv('./models/models.csv', mode='a', header=False)


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)

target_attribute = 'Mouth_Slightly_Open'
hidden_attribute = 'Male'
dataset = datasets.get_dataset(target_attribute)
# dataloader = datasets.get_dataloader(dataset, hidden_attribute, 2000, np.array([0.5, 0.5]))

train_shadow_models(dataset, device, rep=2, lr=0.001, hidden_attribute='Male')
train_shadow_models_test(dataset, device, rep=2, lr=0.001, hidden_attribute='Male')
train_shadow_models_valid(dataset, device, rep=2, lr=0.001, hidden_attribute='Male')
