import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os

import datasets
import models
import utils

def train_shadow_model(dataset, model_type, args, epochs, lr, hidden_attribute, class_distribution, device, size=2000, filename='test'):
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

    path = os.path.join(args.models_dir, f'{filename}.pth')
    torch.save(net.state_dict(), path)


def train_shadow_models(dataset, model_type, args, device, rep=1500, lr=0.001, hidden_attribute='Male', split="train"):
    print(f'\nStarting training of {rep} shadow models on {device} - lr={lr}')
    threshold = 70 #70%
    data = {'model': [], 'property_dist': [], 'split': [], 'architecture': []}

    if not os.path.isfile(args.csv):
        df = pd.DataFrame()
    else:
        df = pd.read_csv(args.csv)
    
    if split == "train":
        split_code = 0
    elif split == "test":
        split_code = 1
    elif split == "valid":
        split_code = 2
    else:
        assert True, "invalid split"

    for model in range(len(df), len(df) + rep, 2):
        # target property p is whether dataset is composed of more than 70% of male images
        p_true = np.random.randint(threshold, 101)
        p_false = np.random.randint(0, threshold)
        p_true_distribution = np.array([100-p_true, p_true])/100
        p_false_distribution = np.array([100-p_false, p_false])/100

        try: # try statements to allow training-interuption and still save the results
            # Train model with property p
            print(f'{model} - p=True - d={p_true_distribution} - Training...')
            filename = f'{model+args.start_from}'
            train_shadow_model(dataset=dataset,
                               model_type=model_type,
                               args=args,
                               epochs=args.epochs,
                               lr=lr,
                               hidden_attribute=hidden_attribute,
                               class_distribution=p_true_distribution,
                               device=device,
                               size=2000,
                               filename=filename)
            data['model'].append(f'{filename}.pth')
            data['property_dist'].append(p_true/100)
            data['split'].append(split_code)
            data['architecture'].append(model_type)
        except:
            break

        try:
            # Train model without property p
            print(f'{model+1} - p=False - d={p_false_distribution} - Training...')
            filename = f'{model+1+args.start_from}'
            train_shadow_model(dataset=dataset,
                               model_type=model_type,
                               args=args,
                               epochs=args.epochs,
                               lr=lr,
                               hidden_attribute=hidden_attribute,
                               class_distribution=p_false_distribution,
                               device=device,
                               size=2000,
                               filename=filename)
            data['model'].append(f'{filename}.pth')
            data['property_dist'].append(p_false/100)
            data['split'].append(split_code)
            data['architecture'].append(model_type)
        except:
            break

    new_df = pd.DataFrame(data).set_index(pd.Index(range(len(df), len(df)+len(data['model']))))
    new_df.to_csv(args.csv, mode='a', header=True)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="space separated list of model architectures, can be a1 to a9")
    parser.add_argument("--csv", type=str, default="template_output.csv", help="path to csv files to store info about shadow models")
    parser.add_argument("--models_dir", type=str, default="./models", help="path to the directory that will store the states of the shadow models.")
    parser.add_argument("--start_from", type=int, default=0, help="index to start from when naming the shadow models")
    parser.add_argument("--epochs", type=int, default=30, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--size_train_shadow", type=int, default=1500, help="number of shadow models to create in the training set.")
    parser.add_argument("--size_test_shadow", type=int, default=200, help="number of shadow models to create in the test set.")
    parser.add_argument("--size_valid_shadow", type=int, default=100, help="number of shadow models to create in the validation set.")
    parser.add_argument("--cuda", action="store_true", default=False, help="use GPU.")

    args = parser.parse_args()


    device = torch.device("cuda:0" if args.cuda else "cpu")

    attribute = 'Mouth_Slightly_Open'
    hidden_attribute = 'Male'
    dataset = datasets.get_dataset(attribute)

    models = args.models
    assert all(model in utils.TYPES_OF_MODEL for model in models)

    for model_t in models:
        print("---- Architecture", model_t, " ----")
        train_shadow_models(dataset, model_t, args, device, rep=args.size_train_shadow, lr=args.lr, hidden_attribute=hidden_attribute, split="train")
        train_shadow_models(dataset, model_t, args, device, rep=args.size_valid_shadow, lr=args.lr, hidden_attribute=hidden_attribute, split="valid")
        train_shadow_models(dataset, model_t, args, device, rep=args.size_test_shadow, lr=args.lr, hidden_attribute=hidden_attribute, split="test")


if __name__ == "__main__":
    main()

