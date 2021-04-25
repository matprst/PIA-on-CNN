import os
import torch
import torch.nn as nn
import torch.optim as optim

import datasets
import models
import utils


def train_attack(dataloader, model, args, device, filename='test'):
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
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
    path = os.path.join(args.attack_models_dir, f"{filename}.pth")
    torch.save(model.state_dict(), path)


def train_all_attacks(args, device):
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

    for attack in attack_models:
        for target_model in architectures:
            for rep in range(args.reps):
                shadow_dataset = datasets.ShadowDataset(args.csv, args.shadow_models_dir, split='train', architecture=str(target_model))
                dataloader = DataLoader(shadow_dataset, batch_size=32, shuffle=True)

                params = utils.number_param(target_model)
                attack_model = attack(in_dim=params).to(device)
                filename = f'{attack_model}-{target_model}-{rep}'
                print(filename)
                train_attack(dataloader, attack_model, args=args, device=device, filename=filename)



def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="template_output.csv", help="path to csv file that contains the info about shadow models.")
    parser.add_argument("--shadow_models_dir", type=str, default="./models/shadow_models/", help="path to the directory that stores the states of the shadow models.")
    parser.add_argument("--attack_models_dir", type=str, default="./models/attack_models/", help="path to the directory that stores the states of the attack models.")
    parser.add_argument("--epochs", type=int, default=10, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--reps", type=int, default=30, help="number of attack models to train for each shadow architecture.")
    parser.add_argument("--cuda", action="store_true", default=False, help="use GPU.")

    args = parser.parse_args()


    device = torch.device("cuda:0" if args.cuda else "cpu"))
    train_all_attacks(args, device)


if __name__ == "__main__":
    main()
