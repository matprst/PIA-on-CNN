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
import os

import models
import utils


def train_attack(model, epoch=10, filename='test'):
    tic = time.time()
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
    # print(f'{rep} - lr={lr} - Training...')
    losses = []
    for epoch in range(epoch):
        running_loss = 0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs).to(device)

            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'[{epoch+1}] loss: {running_loss/(i+1):.3f}')
        # losses.append(running_loss/(i+1))
    tac = time.time()
    print(tac - tic)
    PATH = f'./models/attack_models/{filename}.pth'
    torch.save(model.state_dict(), PATH)


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
    models.AttackNet2,
    models.AttackNet3,
    models.AttackNet4,
    models.AttackNet5
]

EPOCHS = 2
REPS = 2
for attack in attack_models:
    for target_model in architectures:
        for rep in range(REPS):
            shadow_dataset = datasets.ShadowDataset('./models/models.csv', './models/', split='train', architecture=str(target_model))
            dataloader = DataLoader(shadow_dataset, batch_size=32, shuffle=True)

            params = utils.number_param(target_model)
            attack_model = attack(in_dim=params).to(device)
            filename = f'{attack_model}-{target_model}-{rep}'
            print(filename)
            train_attack(attack_model, EPOCHS, filename)
