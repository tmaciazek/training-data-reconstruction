# Vulnerability of Transfer-Learned Neural Networks to Data Reconstruction Attacks in Small-Data Regime

This repository is the official implementation of [Vulnerability of Transfer-Learned Neural Networks to Data Reconstruction Attacks in Small-Data Regime](https://arxiv.org/abs/). 

## Requirements

To install requirements (ideally in a separate environment):

```setup
pip install -r requirements.txt
```

Next, make sure to setup the relevant datasets so that they can be used by Torchvision:

* [EMNIST](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.EMNIST.html) and [MNIST](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html#mnist)
* [CIFAR-100](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR100.html) and [CIFAR-10](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html)
* [CelebA](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CelebA.html)

The files [pretrain_conf_MNIST.yml](/config_data/pretrain_conf_MNIST.yml), [pretrain_conf_CIFAR.yml](/config_data/pretrain_conf_CIFAR.yml) and [pretrain_conf_CelebA.yml](/config_data/pretrain_conf_CelebA.yml) contain the entry `DataRoot` which is currently set to `.` and should be changed if the root directory for the respective datasets is different.
  

## Reproducing the reconstructor NNs

To reproduce the reconstructor NNs, one needs to follow the steps below. 

 ### 1. (Optional) Base model pre-training
 Run the following command, depending on which experiment you aim to reproduce.

```
python pre_training.py --experiment_id=<experiment_id>
```
where `<experiment_id>` is one of `MNIST, CIFAR, CelebA`. You may skip this pre-training step and just use the models included in the folder [models_pretrained](/models_pretrained).
 
### 2. Deep features extraction 
 Run the following command, depending on which experiment you aim to reproduce.

```
python generate_deep_features.py --data_id=<data_id> --model=<model_filename>
```
where `<data_id>` is one of `MNIST, CIFAR10, CIFAR100, CelebA` and `<model_filename>` is the pre-trained model from which we extract the deep features -- one of `VGGtiny_classifier_EMNIST_50EP.pth, EfficientNetB0_CIFAR100_200EP.pth, WideResNet50_CelebA_Attributes_20EP.pth`. This will save the deep features for MNIST, CIFAR-10 and CelebA to the folder [deep_features_data](/deep_features_data). For CelebA this will also save the CelebA images in the $$64\times 64$$-resolution to the folder [celeba_img64](/celeba_img64).

To generate the deep features data needed to reproduce all the experiments, run the following commands.
```
python generate_deep_features.py --data_id=MNIST --model=VGGtiny_classifier_EMNIST_50EP.pth
```
Can be done within seconds on a laptop.
```
python generate_deep_features.py --data_id=CIFAR10 --model=EfficientNetB0_CIFAR100_200EP.pth
```
Took $$90$$ seconds on a GeForce RTX 3090 GPU.
```
python generate_deep_features.py --data_id=CIFAR100 --model=EfficientNetB0_CIFAR100_200EP.pth
```
Took $$90$$ seconds on a GeForce RTX 3090 GPU.
```
python generate_deep_features.py --data_id=CelebA --model=WideResNet50_CelebA_Attributes_20EP.pth
```
Took $$XX$$ minutes on a GeForce RTX 3090 GPU.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
