# Property Inference Attacks on Convolutional Neural Networks:Influence and Implications of Target Model’s Complexity
Authors: Mathias Parisot, Balazs Pejo, and Dayana Spagnuelo


This repository is the official implementation of [Property Inference Attacks on Convolutional Neural Networks:Influence and Implications of Target Model’s Complexity](https://arxiv.org/abs/2104.13061). 

## Requirements

This repository was tested using Python 3 only, please make sure not to use Python 2. To install the dependencies:

```setup
pip install -r requirements.txt
```

## Downloading CelebA

In case you want to train and evaluate the shadow models, you will also need to download [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). For some reasons, downloading CelebA via Pytorch didn't work, so here is how to download CelebA manualy:

- on the CelebA [webpage](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), got to their Google drive directory. The disposition of the directory is as follow:
```
Img/
Anno/
Eval/
README.txt```

- go to `Img/` and download the zip file `img_align_celeba.zip`
- go to `Ano/` and download all the `.txt` files, there should be 5
- go to `Eval/` and doanload `list_eval_partition.txt`
- Move all the downloaded files into `data/celeba` and extract `img_align_celeba.zip`. After that, `data/celeba` should look as follow:
```
img_align_celeba/
identity_CelebA.txt
img_align_celeba.zip
list_attr_celeba.txt
list_bbox_celeba.txt
list_eval_partition.txt
list_landmarks_align_celeba.txt
list_landmarks_celeba.txt
```

Once this is done, you can follow the next sections to train and test the shadow and attack models.

## Training

To train the shadow models (architecture 1, 2, 3, and 4 from the paper), run this command:

```train
python train_shadow.py --models a1 a2 a3 a4 --csv template_output.csv 
```

To train the attack models, run this command:

```train
python train_attack.py --csv template_output.csv 
```

For more specific parameters, please check the command line arguments of the respective files.

## Evaluation

After training the shadow models, you can evaluate their performance on [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) by running:

```eval
python test_shadow.py --shadow_csv template_output.csv
```

After training the attack models, you can evaluate their performance on the shadow model test dataset by running:

```eval
python test_attack.py --csv template_output.csv
```
