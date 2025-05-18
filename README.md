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
Took $$12$$ seconds on Apple M4 Pro.
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
Took $$30$$ minutes on a GeForce RTX 3090 GPU.

### 3. Shadow model training

#### Non-private training

Run the following command.

```
shadow_model_training.py --data_id=<data_id> --split=<split> --permutation_seed=<seed> --models_per_seed=<models_per_seed>
```
where `<data_id>` is one of `MNIST, CIFAR10, CIFAR100, CelebA`. Set `<split>` to `train/test` to generate training/validation shadow models respectively. The integer `<seed>` determines the random seed that has been used to permute the data and create the training sets of the shadow models. Each seed generates `<models_per_seed>` shadow models. For instance, if `<models_per_seed>=5000`, then to generate $$2.56\times 10^6$$ training shadow models one needs to use `<seed>` from the range $$0,1,\dots,511$$. Scripts with different seeds are  meant to run in parallel on many CPUs using for instance array jobs in `SLURM`. For instance, to geneate $$2.56\times 10^6$$ training shadow models for CIFAR10 one would run a `SLURM` job script with `#SBATCH --array=0-511` and
```
python shadow_model_training.py --data_id=CIFAR10 --split=train --permutation_seed=${SLURM_ARRAY_TASK_ID} --models_per_seed=5000
```
The test shadow models would be generated as follows.
```
python shadow_model_training.py --data_id=CIFAR10 --split=test --permutation_seed=0 --models_per_seed=1000
```
We use just a single seed (default $$0$$) for validation shadow models.

The files [shadow_conf_multiclass_MNIST.yml](/config_data/shadow_conf_multiclass_MNIST.yml), [shadow_conf_binary_MNIST.yml](/config_data/shadow_conf_binary_MNIST.yml), [shadow_conf_multiclass_CIFAR.yml](/config_data/shadow_conf_multiclass_CIFAR.yml), [shadow_conf_binary_CIFAR.yml](/config_data/shadow_conf_binary_CIFAR.yml), [shadow_conf_CelebA.yml](/config_data/shadow_conf_CelebA.yml) contain all the hyperparameter configurations that are needed to recreate all the main experiments as well as experiments from the section `3.2 Factors Affecting Reconstruction` in the paper. Note that this will need the manual change of the variables such as `CLASS_SIZE` (1 through 4) , `TRAINING_EPOCHS` (1 through 512), `INIT_STD' (0.0002 to 0.2 in the logarighmic scale), `OPTIMIZER` (SGD/Adam). To recreate the OOD experiment, one needs to generate CIFAR100 training shadow models, train the reconstructor NN and test it on the CIFAR10 validation shadow models.

Note that you can use `--filename_appendix=<app>` to add more info to the saved weights filename (currently default `<app>=classifier`).

The seed ranges and the `models_per_seed` configurations needed to generate $$2.56\times 10^6$$ training shadow models in the different experiments are summarized in the table below. 

| `<data_id>`, N        | seed range  | `models_per_seed`, train split| `models_per_seed`, test split|
| ------------------ |---------------- | -------------- |-------------- |
| MNIST, N=10   |     0-499         |      5120       | 892 |
| MNIST, N=40   |     0-1999         |      1280       | 223 |
| CIFAR, N=10   |     0-511         |      5000       | 1000 |
| CIFAR, N=40   |     0-2047         |      1250       | 250 |
| CelebA, N=10   |     0-124         |      20480       | 10240 |
| CelebA, N=40   |     0-124         |      20480       | 10240 |

The table below shows shadow model generation times obtained for the above argument configurations on Apple M4 Pro chip.

| `<data_id>`, N      | time per seed [s] | 
| ------------------ |---------------- |
| MNIST, N=10   |       23      | 
| MNIST, N=40   |        11      | 
| CIFAR, N=10   |        32     | 
| CIFAR, N=40   |         18    | 
| CelebA, N=10   |       2076       | 
| CelebA, N=40   |        2183     |


#### Private training

### 3. Computing shadow model weights stats

Run the following command

```
python weight_stats.py --filename_header=<header> --seed_range=<seed_range>
```

The `<header>` is the header (the bit excluding the `_train_seed_*.npy` part from the file name) of the filenames containing weights that were produced in point **2** above. The `<seed_range>` is an integer that determines the number of seeds used to produce the shadow models (see the tables in point **2** above).

The script also verifies if the shadow model files corresponding to all the permutation seeds have been saved successfully.

### 4. Reconstructor NN training.

After having generated the shadow models data as well as the corresponding weight statistics it is now time to train the reconstructor NNs. To do this, tun the following scripts.

To train the conditional reconstructor:
```
python cond_reconstructor.py --data_id=<data_id> --filename_header=<header> --seed_range=<seed_range> --rec_name=<rec_name>
```
To train the unconditional reconstructor:
```
python uncond_reconstructor.py --data_id=<data_id> --filename_header=<header> --seed_range=<seed_range> --rec_name=<rec_name>
```
As before, the `<header>` is the header (the bit excluding the `_train_seed_*.npy` part from the file name) of the filenames containing weights that were produced in point **2** above. The `<seed_range>` is an integer that determines the number of seeds used to produce the shadow models (see the tables in point **2** above). The string `<rec_name>` will be the name of the saved reconstructor NN model.

The config files  [reconstruction_conf_MNIST.yml](/config_data/reconstruction_conf_MNIST.yml), [reconstruction_conf_CIFAR.yml](/config_data/reconstruction_conf_CIFAR.yml), [reconstruction_conf_CIFAR.yml](/config_data/reconstruction_conf_CIFAR.yml) contain the following entries.



* `MSE_THRESHOLD` - the nearest neighbour MSE threshold used for reconstruction TPR estimation (see Section `2. Reconstruction Robustness Measures` of the paper)
* `CLASS_SIZE` - the number of training examples per class in the training sets of the classifier shadow models -- this must be compatible with the class sizes used to generate the shadow model weights contained in the files determined by `--filename_header`
* `GEN_SIZE` - reconstructor NN internal size parameter
* `VAL_BATCH_SIZE` - (technical) the batch size used for validating the reconstructor NN
* `DATA_MEM_LIMIT` - rough amount of memory we can allocate for the pre-loaded reconstructor trainig data
* `SEEDS_PER_EPOCH` - associated with the `DATA_MEM_LIMIT` - number of training data seeds that are pre-loaded at the beginning of each epoch
* `BATCH_SIZE` - the batch size used for reconstructor NN training
* `LR` - the learning rate used for reconstructor NN training
* `TRAINING_EPOCHS` - the number of epochs of reconstructor NN training (checkpoints are saved automatically every 500K gradient steps)
* `DataRoot` - the root directory for the relevant image dataset


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
