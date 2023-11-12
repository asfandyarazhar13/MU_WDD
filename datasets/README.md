# Datasets

Each `.zip` file contains distilled tensor images (with their corresponding labels) of popular machine learning benchmark datasets. 

Namely:
* CIFAR-10
* CIFAR-100
* ImageNet
* Tiny ImageNet
* MNIST
* SVHN
* ANIMAL-10N.

When the files are unzipped directories will have the following naming conventions: 

1. `<DATASET>_ipc<N>`: Corresponds to a dataset which has `N` distilled images per class.
2. `<DATASET>_ipc<N>_zca`: Corresponds to a dataset which has `N` distilled images per class where [ZCA whitening](http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/) is applied.


### Some notes: 
1. The for the sake of simplicity, the distilled images are extracted using some of the best hyperparameters detailed in the paper [_Dataset Distillation by Matching Training Trajectories_](https://arxiv.org/abs/2203.11932). See table below.
