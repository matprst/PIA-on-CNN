# Property Inference Attacks on Convolutional Neural Networks:Influence and Implications of Target Model’s Complexity
Authors: Mathias Parisot, Balazs Pejo, and Dayana Spagnuelo


This repository is the official implementation of [Property Inference Attacks on Convolutional Neural Networks:Influence and Implications of Target Model’s Complexity](https://arxiv.org/abs/2104.13061). 

## Requirements

This repository was tested using Python 3 only, please make sure not to use Python 2. To install the dependencies:

```setup
pip install -r requirements.txt
```

In case you want to train and evaluate the shadow models, you will also need to download [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Please refer to their webpage for the instructions.

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
