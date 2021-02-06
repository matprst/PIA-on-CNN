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
import utils

import traceback

def train_shadow_model(dataset, model_type, epochs, lr, hidden_attribute, class_distribution, device, size=2000, filename='test'):
    dataloader = datasets.get_dataloader(dataset, hidden_attribute, size, class_distribution)
    net = utils.get_model(model_type).to(device)

    tic = time.time()
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        running_loss = 0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data

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

    # path = f'models/test/{filename}.pth'
    path = f'models/shadow_models/eyeglasses/{filename}.pth'
    torch.save(net.state_dict(), path)


def train_shadow_models(dataset, model_type, args, device, rep=1500, lr=0.001, hidden_attribute='Male'):
    print(f'\nStarting training of {rep} shadow models on {device} - lr={lr}')
    threshold = 70 #70%
    data = {'model': [], 'glasses_dist': [], 'split': [], 'architecture': []}
    # df = pd.read_csv('models/test/models.csv')
    # df = pd.read_csv('models/shadow_models/eyeglasses/models-glasses.csv')
    df = pd.read_csv(args.csv)

    for model in range(len(df), len(df) + rep, 2):
        # target property p is whether dataset is composed of more than 70% of male images
        p_true = np.random.randint(threshold, 101)
        p_false = np.random.randint(0, threshold)
        p_true_distribution = np.array([100-p_true, p_true])/100
        p_false_distribution = np.array([100-p_false, p_false])/100

        try:
            # Train model with property p
            print(f'{model} - p=True - d={p_true_distribution} - Training...')
            filename = f'{model+args.start_from}'
            train_shadow_model(dataset=dataset,
                               model_type=model_type,
                               epochs=args.epochs,
                               lr=lr,
                               hidden_attribute=hidden_attribute,
                               class_distribution=p_true_distribution,
                               device=device,
                               size=2000,
                               filename=filename)
            data['model'].append(f'{filename}.pth')
            data['glasses_dist'].append(p_true/100)
            data['split'].append(0)
            data['architecture'].append(model_type)
        except:
            traceback.print_exc(file=sys.stdout)
            break

        try:
            # Train model without property p
            print(f'{model+1} - p=False - d={p_false_distribution} - Training...')
            filename = f'{model+1+args.start_from}'
            train_shadow_model(dataset=dataset,
                               model_type=model_type,
                               epochs=args.epochs,
                               lr=lr,
                               hidden_attribute=hidden_attribute,
                               class_distribution=p_false_distribution,
                               device=device,
                               size=2000,
                               filename=filename)
            data['model'].append(f'{filename}.pth')
            data['glasses_dist'].append(p_false/100)
            data['split'].append(0)
            data['architecture'].append(model_type)
        except:
            traceback.print_exc(file=sys.stdout)
            break

    new_df = pd.DataFrame(data).set_index(pd.Index(range(len(df), len(df)+len(data['model']))))
    # new_df.to_csv('models/test/models.csv', mode='a', header=False)
    # new_df.to_csv('models/shadow_models/eyeglasses/models-glasses.csv', mode='a', header=False)
    new_df.to_csv(args.csv, mode='a', header=False)

def train_shadow_models_test(dataset, model_type, args, device, rep=200, lr=0.001, hidden_attribute='Male'):
    print(f'\nStarting training of {rep} shadow models test on {device} - lr={lr}')
    threshold = 70
    data = {'model': [], 'glasses_dist': [], 'split': [], 'architecture': []}
    # df = pd.read_csv('models/shadow_models/eyeglasses/models-glasses.csv')
    df = pd.read_csv(args.csv)
    for model in range(len(df), len(df) + rep, 2):
        # target property p is whether dataset is composed of more than 70% of male images
        p_true = np.random.randint(threshold, 101)
        p_false = np.random.randint(0, threshold)
        p_true_distribution = np.array([100-p_true, p_true])/100
        p_false_distribution = np.array([100-p_false, p_false])/100

        try:
            # Train model with property p
            print(f'{model} - p=True - d={p_true_distribution} - Training...')
            filename = f'{model+args.start_from}'
            train_shadow_model(dataset=dataset,
                               model_type=model_type,
                               epochs=args.epochs,
                               lr=lr,
                               hidden_attribute=hidden_attribute,
                               class_distribution=p_true_distribution,
                               device=device,
                               size=2000,
                               filename=filename)
            data['model'].append(f'{filename}.pth')
            data['glasses_dist'].append(p_true/100)
            data['split'].append(1)
            data['architecture'].append(model_type)
        except:
            break

        try:
            # Train model without property p
            print(f'{model+1} - p=False - d={p_false_distribution} - Training...')
            filename = f'{model+1+args.start_from}'
            train_shadow_model(dataset=dataset,
                               model_type=model_type,
                               epochs=args.epochs,
                               lr=lr,
                               hidden_attribute=hidden_attribute,
                               class_distribution=p_false_distribution,
                               device=device,
                               size=2000,
                               filename=filename)
            data['model'].append(f'{filename}.pth')
            data['glasses_dist'].append(p_false/100)
            data['split'].append(1)
            data['architecture'].append(model_type)
        except:
            break

    new_df = pd.DataFrame(data).set_index(pd.Index(range(len(df), len(df)+len(data['model']))))
    # new_df.to_csv('models/shadow_models/eyeglasses/models-glasses.csv', mode='a', header=False)
    new_df.to_csv(args.csv, mode='a', header=False)

def train_shadow_models_valid(dataset, model_type, args, device, rep=100, lr=0.001, hidden_attribute='Male'):
    print(f'\nStarting training of {rep} shadow models valid on {device} - lr={lr}')
    threshold = 70
    data = {'model': [], 'glasses_dist': [], 'split': [], 'architecture': []}
    # df = pd.read_csv('models/shadow_models/eyeglasses/models-glasses.csv')
    df = pd.read_csv(args.csv)

    for model in range(len(df), len(df) + rep, 2):
        # target property p is whether dataset is composed of more than 70% of male images
        p_true = np.random.randint(threshold, 101)
        p_false = np.random.randint(0, threshold)
        p_true_distribution = np.array([100-p_true, p_true])/100
        p_false_distribution = np.array([100-p_false, p_false])/100

        try:
            # Train model with property p
            print(f'{model} - p=True - d={p_true_distribution} - Training...')
            filename = f'{model+args.start_from}'
            train_shadow_model(dataset=dataset,
                               model_type=model_type,
                               epochs=args.epochs,
                               lr=lr,
                               hidden_attribute=hidden_attribute,
                               class_distribution=p_true_distribution,
                               device=device,
                               size=2000,
                               filename=filename)
            data['model'].append(f'{filename}.pth')
            data['glasses_dist'].append(p_true/100)
            data['split'].append(2)
            data['architecture'].append(model_type)
        except:
            break

        try:
            # Train model without property p
            print(f'{model+1} - p=False - d={p_false_distribution} - Training...')
            filename = f'{model+1+args.start_from}'
            train_shadow_model(dataset=dataset,
                               model_type=model_type,
                               epochs=args.epochs,
                               lr=lr,
                               hidden_attribute=hidden_attribute,
                               class_distribution=p_false_distribution,
                               device=device,
                               size=2000,
                               filename=filename)
            data['model'].append(f'{filename}.pth')
            data['glasses_dist'].append(p_false/100)
            data['split'].append(2)
            data['architecture'].append(model_type)
        except:
            break

    new_df = pd.DataFrame(data).set_index(pd.Index(range(len(df), len(df)+len(data['model']))))
    # new_df.to_csv('models/shadow_models/eyeglasses/models-glasses.csv', mode='a', header=False)
    new_df.to_csv(args.csv, mode='a', header=False)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", help="space separated list of model architectures")
    parser.add_argument("--csv", type=str, default="models.csv", help="path to csv files to store info about shadow models")
    parser.add_argument("--start_from", type=int, default=0, help="index to start from when naming the shadow models")
    parser.add_argument("--epochs", type=int, default=30, help="number of training epochs")


    args = parser.parse_args()

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    attribute = 'Male'
    hidden_attribute = 'Eyeglasses'
    dataset = datasets.get_dataset(attribute)

    models = args.models
    assert all(model in utils.TYPES_OF_MODEL for model in models)

    for model_t in models:
        print("---- Architecture", model_t, " ----")
        train_shadow_models(dataset, model_t, args, device, rep=1500, lr=0.001, hidden_attribute=hidden_attribute)
        train_shadow_models_test(dataset, model_t, args, device, rep=200, lr=0.001, hidden_attribute=hidden_attribute)
        train_shadow_models_valid(dataset, model_t, args, device, rep=100, lr=0.001, hidden_attribute=hidden_attribute)



if __name__ == "__main__":
    main()

