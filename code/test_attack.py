import torch

from torch.utils.data import DataLoader

import pandas as pd

import datasets
import glob

import models
import utils


def get_attack(number):
    attacks = {
        1: models.AttackNet1,
        2: models.AttackNet2,
        3: models.AttackNet3,
        4: models.AttackNet4,
        5: models.AttackNet5,
    }
    return attacks[number]


# device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
# for i in range(1, 6):
#     models_files = sorted(glob.glob(f'models/attack_models/full/attack-{i}*.pth'))

#     data = {
#         'model':[],
#         'architecture':[],
#         'num_param':[],
#         'accuracy':[],
#         'precision':[],
#         'recall':[]
#     }

#     for model in models_files:
#         # model name is of the form "{attack_model}-{target_model}-{rep}"
#         filename = model.split('/')[-1]
#         architecture = filename.split('-')[2]

#         shadow_dataset = datasets.ShadowDataset('models/shadow_models/gender/models.csv', 'models/shadow_models/gender/', split='test', architecture=architecture, fcn=False, conv=False, device=device)
#         test_loader = DataLoader(shadow_dataset, batch_size=32, shuffle=True)

#         params = utils.number_param(utils.get_model(architecture), fcn=False, conv=False)
#         print(params)
#         attack_model = get_attack(i)(in_dim=params).to(device)
        
#         device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

#         attack_model.load_state_dict(torch.load(model, map_location=torch.device(device)))

#         acc, prec, recall = utils.evaluate(test_loader, attack_model, device)
#         print(acc, prec, recall)
#         data['model'].append(filename)
#         data['architecture'].append(architecture)
#         data['num_param'].append(params)
#         data['accuracy'].append(acc)
#         data['precision'].append(prec)
#         data['recall'].append(recall)

#     df = pd.DataFrame(data)
#     # df.to_csv('models/attack_models/conv/attack_models_conv.csv', mode='a', header=False)

def test_all_attacks(args, device):
    for i in range(1, 6):
        files = f"{args.attack_models_dir}/attack-{i}*.pth"
        # models_files = sorted(glob.glob(f'models/attack_models/full/attack-{i}*.pth'))
        models_files = sorted(glob.glob(files))

        data = {
            'model':[],
            'architecture':[],
            'num_param':[],
            'accuracy':[],
            'precision':[],
            'recall':[]
        }

        use_only_fcn = False
        use_only_conv = False
        for model in models_files:
            # model name is of the form "{attack_model}-{target_model}-{rep}"
            filename = model.split('/')[-1]
            architecture = filename.split('-')[2]

            # shadow_dataset = datasets.ShadowDataset('models/shadow_models/gender/models.csv', 'models/shadow_models/gender/', split='test', architecture=architecture, fcn=False, conv=False, device=device)
            shadow_dataset = datasets.ShadowDataset(
                args.csv, 
                args.shadow_models_dir, 
                split='test', 
                architecture=architecture, 
                fcn=use_only_fcn, 
                conv=use_only_conv, 
                device=device
            )
            test_loader = DataLoader(shadow_dataset, batch_size=32, shuffle=True)

            params = utils.number_param(utils.get_model(architecture), fcn=use_only_fcn, conv=use_only_conv)
            attack_model = get_attack(i)(in_dim=params).to(device)
            attack_model.load_state_dict(torch.load(model, map_location=torch.device(device)))

            acc, prec, recall = utils.evaluate(test_loader, attack_model, device)
            print(acc, prec, recall)
            data['model'].append(filename)
            data['architecture'].append(architecture)
            data['num_param'].append(params)
            data['accuracy'].append(acc)
            data['precision'].append(prec)
            data['recall'].append(recall)

        df = pd.DataFrame(data)
        # df.to_csv('models/attack_models/conv/attack_models_conv.csv', mode='a', header=False)
        df.to_csv(args.results_csv, mode='a', header=False)

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="template_output.csv", help="path to csv file that contains the info about shadow models.")
    parser.add_argument("--shadow_models_dir", type=str, default="./models/shadow_models/", help="path to the directory that stores the states of the shadow models.")
    parser.add_argument("--attack_models_dir", type=str, default="./models/attack_models/", help="path to the directory that stores the states of the attack models.")
    parser.add_argument("--results_csv", type=str, default="./attacks_performance.csv", help="csv file where the performance of the attack models will be output.")
    parser.add_argument("--cuda", action="store_true", default=False, help="use GPU.")

    args = parser.parse_args()


    device = torch.device("cuda:0" if args.cuda else "cpu")
    test_all_attacks(args, device)


if __name__ == "__main__":
    main()
