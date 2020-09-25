import time
import PIL
import os, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
import os

import models
import utils


def train_attack(train_loader, val_loader, model, criterion, optimizer, epoch=10, filename='test'):
    tic = time.time()
    # criterion = nn.MSELoss(reduction='sum')
    # optimizer = optim.Adam(attack_model.parameters(), lr=lr)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    # print(f'{rep} - lr={lr} - Training...')
    losses = []
    validate_attack(val_loader, model, device)
    for epoch in range(epoch):
        running_loss = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs).to(device)

            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # print(f'[{epoch+1}] loss: {running_loss/(i+1):.3f}')
        # losses.append(running_loss/(i+1))
        acc, prec, recall = validate_attack(val_loader, model, device)
    # print(f'\tAccuracy: {acc} %')
    # print(f'\tPrecision: {prec} %')
    # print(f'\tRecall: {recall} %')
    tac = time.time()
    # print(tac - tic)
    # PATH = f'./models/attack_models/{filename}.pth'
    # torch.save(model.state_dict(), PATH)
    return acc

def validate_attack(dataloader, model, device):
    return utils.evaluate(dataloader, attack_model, device)



device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)

architectures = [
    models.Net1(),
    models.Net2(),
    models.Net3(),
    models.Net4(),
    models.Net5(),
    models.Net6(),
    models.Net7(),
    models.Net8(),
    models.Net9(),
]

attack_models = [
    models.AttackNet1,
    # models.AttackNet2,
    # models.AttackNet3,
    # models.AttackNet4,
    # models.AttackNet5,
    # models.AttackNet6
]

EPOCHS = 10
REPS = 10
lrs = [0.005, 0.001, 0.0005]
batch_sizes = [16, 32, 64]
losses = ['MSE', 'L1']
optimizers = ['adam', 'SGD']
activations = ['relu', 'tanh', 'sigmoid']


data = {
    'target-model': [],
    'lr': [],
    'batch-size': [],
    'loss-function': [],
    'optimizer': [],
    'rep': [],
    'activation': [],
    'accuracy': []
}

tic = time.time()
try:
    for rep in range(6, REPS):
        for attack in attack_models:
            for target_model in architectures:
                for lr in lrs:
                    for batch_size in batch_sizes:
                        for loss_name in losses:
                            if loss_name == 'MSE':
                                loss = nn.MSELoss
                            elif loss_name == 'L1':
                                loss = nn.L1Loss
                            else:
                                raise Exception('wrong loss name')

                            for opt_name in optimizers:
                                if opt_name == 'adam':
                                    opt = optim.Adam
                                elif opt_name == 'SGD':
                                    opt = optim.SGD
                                else:
                                    raise Exception('wrong optimizer name')

                                for activation in activations:
                                    print(f"target_model={str(target_model)}, {lr=}, {batch_size=}, {loss_name=}, {opt_name=}, {activation=}, {rep=}")
                                    shadow_dataset_train = datasets.ShadowDataset('models/shadow_models/models.csv', 'models/shadow_models/', split='train', architecture=str(target_model))
                                    shadow_dataset_val = datasets.ShadowDataset('models/shadow_models/models.csv', 'models/shadow_models/', split='valid', architecture=str(target_model))
                                    dataloader_train = DataLoader(shadow_dataset_train, batch_size=batch_size, shuffle=True)
                                    dataloader_val = DataLoader(shadow_dataset_val, batch_size=batch_size, shuffle=True)

                                    params = utils.number_param(target_model)
                                    attack_model = attack(in_dim=params, activation=activation).to(device)
                                    filename = f'{attack_model}-{target_model}-{rep}'
                                    # print(filename)

                                    criterion = loss(reduction='sum')
                                    optimizer = opt(attack_model.parameters(), lr=lr)
                                    
                                    acc = train_attack(dataloader_train, dataloader_val, attack_model, criterion, optimizer, EPOCHS, filename)
                                    # validate_attack(dataloader_val, attack_model)
                                    data['target-model'].append(str(target_model))
                                    data['lr'].append(lr)
                                    data['batch-size'].append(batch_size)
                                    data['loss-function'].append(loss_name)
                                    data['optimizer'].append(opt_name)
                                    data['rep'].append(rep)
                                    data['activation'].append(activation)
                                    data['accuracy'].append(acc)
except:
    df = pd.DataFrame.from_dict(data)
    df.to_csv('./attack_tuning3.csv', mode='a', header=False)
    print(df)
    tac = time.time()
    print(tac - tic)
    sys.exit()
                                
df = pd.DataFrame.from_dict(data)
df.to_csv('./attack_tuning3.csv', mode='a', header=False)
print(df)
tac = time.time()
print(tac - tic)
