# Datasets

Each `.zip` file contains distilled tensor images (with their corresponding labels) of popular machine learning benchmark datasets. 

Namely:
* CIFAR-10
* CIFAR-100

When the files are unzipped directories will have the following naming conventions: 

1. `<DATASET>_ipc<N>`: Corresponds to a dataset which has `N` distilled images per class.
2. `<DATASET>_ipc<N>_zca`: Corresponds to a dataset which has `N` distilled images per class where [ZCA whitening](http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/) is applied.


### Some notes: 
1. For the sake of simplicity, the distilled images are extracted using some of the best hyperparameters detailed in the papers: 
* _[Dataset Distillation by Matching Training Trajectories](https://arxiv.org/abs/2203.11932)_
* _[Scaling Up Dataset Distillation to ImageNet-1K with Constant Memory](https://arxiv.org/abs/2211.10586)_
