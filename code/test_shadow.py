import torch

import pandas as pd

import models
import datasets
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

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)

attribute = 'Male'
# attribute = 'Mouth_Slightly_Open'
train_set = datasets.get_dataset(attribute, split='train')
train_loader = datasets.get_dataloader(train_set, property=None, size=None, class_proportions=None, batch_size=64)
test_set = datasets.get_dataset(attribute, split='test')
test_loader = datasets.get_dataloader(test_set, property=None, size=None, class_proportions=None, batch_size=64)

data_iter = iter(test_loader)
images, labels = data_iter.next()

# df_models = pd.read_csv('models/shadow_models/models.csv', index_col=0)
df_models = pd.read_csv('models/test/models.csv', index_col=0)
print(df_models)
data = {'model': [], 'architecture': [], 'accuracy_train': [], 'precision_train': [], 'recall_train': [], 'accuracy_test': [], 'precision_test': [], 'recall_test': []}
architecture_ids = ('a'+str(i) for i in range(1, 10))

for architecture in architecture_ids:
    df_models_architecture = df_models[df_models['architecture'] == architecture]['model'].values
    for filename in df_models_architecture:
        # path = f'models/shadow_models/{filename}'
        path = f'models/test/{filename}'
        print(path)
        model = get_model(architecture)()
        model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        a_train, p_train, r_train = utils.evaluate(train_loader, model.to(device), device)
        a_test, p_test, r_test = utils.evaluate(test_loader, model.to(device), device)
        data['precision_train'].append(f'{p_train:.3f}')
        data['accuracy_train'].append(f'{a_train:.3f}')
        data['recall_train'].append(f'{r_train:.3f}')

        data['precision_test'].append(f'{p_test:.3f}')
        data['accuracy_test'].append(f'{a_test:.3f}')
        data['recall_test'].append(f'{r_test:.3f}')
        data['model'].append(filename)
        data['architecture'].append(architecture)
    new_df = pd.DataFrame(data)
    # new_df.to_csv('models/shadow_models/models_perf.csv', mode='a', header=False)
    new_df.to_csv('models/test/models_perf.csv', mode='a', header=False)
    # data = {'model': [], 'architecture': [], 'accuracy': [], 'precision': [], 'recall': []}
    data = {'model': [], 'architecture': [], 'accuracy_train': [], 'precision_train': [], 'recall_train': [], 'accuracy_test': [], 'precision_test': [], 'recall_test': []}
