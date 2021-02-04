import torch
import models

TYPES_OF_MODEL = {
        "a1": models.Net1(),
        "a2": models.Net2(),
        "a3": models.Net3(),
        "a4": models.Net4(),
        "a5": models.Net5(),
        "a6": models.Net6(),
        "a7": models.Net7(),
        "a8": models.Net8(),
        "a9": models.Net9(),
        "a10": models.Net10(),
    }

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

def get_model(model_type):
    '''
    Return a pytorch module corresponding to the model_type string.
    '''
    assert model_type in TYPES_OF_MODEL, "wrong model type"
    return TYPES_OF_MODEL[model_type]


def evaluate(dataloader, net, device):
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
    # print(f'\tAccuracy: {accuracy} %')
    # print(f'\tPrecision: {precision} %')
    # print(f'\tRecall: {recall} %')
    return accuracy, precision, recall

def number_param(model, fcn=False, conv=False):
    '''
    Return the number of weights/parameters of the given model. Set either fcn or conv 
    to True to only the the fully connected layers or the convolution layers respectively.
    '''
    s = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            if fcn:
                if 'fc' in n:
                    s += p.numel()
            elif conv:
                if 'conv' in n:
                    s += p.numel()
            else:
                s += p.numel()
    return s