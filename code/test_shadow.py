import torch

import pandas as pd

import models
import datasets


def get_eval_metrics(predicted, actual):
    '''
    Return the true positive, true negative, false positive and false negative
    counts given the predicted and actual values.
    '''
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for p, a in zip(predicted, actual):
        if p == 1:
            if a == 1:
                tp += 1
            else:
                fp += 1
        elif p == 0:
            if a == 1:
                fn += 1
            elif a == 0:
                tn += 1
    return tp, tn, fp, fn

def evaluate(path, dataloader, net):
    '''
    Evaluate the model on the dataloader and return the accuracy, precision, and recall.
    '''
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = net(images).to(device)
            
            predicted = (outputs.view(-1)>0.5).float()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            tp, tn, fp, fn = get_eval_metrics(predicted, labels)
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn

    if total_tp + total_fp != 0:
        precision = total_tp / (total_tp + total_fp)
    else: precision = '//'
    recall = total_tp / (total_tp + total_fn)
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
    print(f'\tAccuracy: {accuracy} %')
    print(f'\tPrecision: {precision} %')
    print(f'\tRecall: {recall} %')
    return accuracy, precision, recall


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

test_set = datasets.get_dataset('Mouth_Slightly_Open', split='test')
test_loader = datasets.get_dataloader(test_set, property=None, size=None, class_proportions=None, batch_size=64)

data_iter = iter(test_loader)
images, labels = data_iter.next()

df_models = pd.read_csv('models/models.csv', index_col=0)
print(df_models)
data = {'model': [], 'architecture': [], 'accuracy': [], 'precision': [], 'recall': []}
architecture_ids = ('a'+str(i) for i in range(1, 10))

for architecture in architecture_ids:
    df_models_architecture = df_models[df_models['architecture'] == architecture]['model'].values
    for filename in df_models_architecture:
        path = f'./models/{filename}'
        print(path)
        model = get_model(architecture)()
        model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        a, p, r = evaluate(path, test_loader, model.to(device))
        data['precision'].append(f'{p:.3f}')
        data['accuracy'].append(f'{a:.3f}')
        data['recall'].append(f'{r:.3f}')
        data['model'].append(filename)
        data['architecture'].append(architecture)
    new_df = pd.DataFrame(data)
    # new_df.to_csv('./models/models_perf.csv', mode='a', header=False)
    data = {'model': [], 'architecture': [], 'accuracy': [], 'precision': [], 'recall': []}
