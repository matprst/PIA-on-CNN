import torch

from torch.utils.data import DataLoader

import pandas as pd

import datasets
import glob

import models
import utils


def get_model(architecture):
    architectures = {
        'a1': models.Net1,
        'a2': models.Net2,
        'a3': models.Net3,
        'a4': models.Net4,
        'a5': models.Net5,
        'a6': models.Net6,
        'a7': models.Net7,
        'a8': models.Net8,
        'a9': models.Net9
    }
    return architectures[architecture]

def get_attack(number):
    attacks = {
        1: models.AttackNet1,
        2: models.AttackNet2,
        3: models.AttackNet3,
        4: models.AttackNet4,
        5: models.AttackNet5,
    }
    return attacks[number]


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
for i in range(1, 6):
    models_files = sorted(glob.glob(f'./models/attack_models_conv/attack-{i}*.pth'))

    data = {
        'model':[],
        'architecture':[],
        'num_param':[],
        'accuracy':[],
        'precision':[],
        'recall':[]
    }

    for model in models_files:
        filename = model.split('/')[-1]
        architecture = filename.split('-')[2]
        number = int(filename.split('-')[4].split('.')[0])
        print(filename, architecture, number)
        
        shadow_dataset = datasets.ShadowDataset('./models/models.csv', './models/', split='test', architecture=architecture, fcn=False, conv=True, device=device)
        test_loader = DataLoader(shadow_dataset, batch_size=32, shuffle=True)

        params = utils.number_param(get_model(architecture)(), fcn=False, conv=True)
        print(params)
        attack_model = get_attack(i)(in_dim=params).to(device)
        
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        attack_model.load_state_dict(torch.load(model, map_location=torch.device(device)))

        acc, prec, recall = utils.evaluate(test_loader, attack_model, device)
        data['model'].append(filename)
        data['architecture'].append(architecture)
        data['num_param'].append(params)
        data['accuracy'].append(acc)
        data['precision'].append(prec)
        data['recall'].append(recall)

    df = pd.DataFrame(data)
    # df.to_csv('./models/attack_models_conv/attack_models_conv.csv', mode='a', header=False)
