import torch
import resource
import numpy as np
import lpips
import argparse
import yaml
import sys, os
import time

import torch.nn.functional as F

import torch.utils.data as torch_data_utils

from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from torchsummary import summary


import matplotlib.pyplot as plt

from data_utils import *
from models import Reconstructor

print("CUDA available?\t", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def show_tensor_images(
    image_tensor, num_images=25, size=(1, 32, 32), nrow=5, save=True, img_name=None
):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    """
    plt.clf()
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu().squeeze()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if save:
        plt.savefig(img_name + ".pdf")


"""
	Defining different training loss functions.
"""


def set_softmin_mse_mae(outputs, sets, alpha=100.0):
    assert sets.shape[0] == outputs.shape[0]
    assert sets.shape[-3] == outputs.shape[-3]

    N_IMG = sets.shape[1]
    RES = sets.shape[-1]
    B = sets.shape[0]
    CH = sets.shape[-3]
    se = torch.mean(torch.square(outputs - sets), dim=(-3, -2, -1))
    ae = torch.mean(torch.abs(outputs - sets), dim=(-3, -2, -1))
    err = 0.5 * (se + ae)

    return torch.mean(torch.sum(err * F.softmax(-alpha * err, dim=1), dim=1))


def mse_mae(outputs, sets):
    assert sets.shape[0] == outputs.shape[0]
    assert sets.shape[-3] == outputs.shape[-3]

    se = torch.mean(torch.square(outputs - sets), dim=(-3, -2, -1))
    ae = torch.mean(torch.abs(outputs - sets), dim=(-3, -2, -1))
    err = 0.5 * (se + ae)

    return torch.mean(err)


def set_softmin_mse_mae_lpips(outputs, sets, alpha=100.0):
    assert sets.shape[0] == outputs.shape[0]
    assert sets.shape[-3] == outputs.shape[-3]

    N_IMG = sets.shape[1]
    RES = sets.shape[-1]
    B = sets.shape[0]
    CH = sets.shape[-3]
    se = torch.mean(torch.square(outputs - sets), dim=(-3, -2, -1))
    ae = torch.mean(torch.abs(outputs - sets), dim=(-3, -2, -1))
    se_ae = 0.5 * (se + ae)

    sets_res = sets.reshape(-1, CH, RES, RES)
    out_repeat = outputs.repeat(1, N_IMG, 1, 1, 1).reshape(-1, CH, RES, RES)
    lpips_batch = lpips_loss_fn.forward(out_repeat, sets.reshape(-1, CH, RES, RES))
    lpips_table = lpips_batch.reshape(B, N_IMG)

    assert lpips_table.shape == se_ae.shape
    err = se_ae + lpips_table

    return torch.mean(torch.sum(err * F.softmax(-alpha * err, dim=1), dim=1))


def mse_mae_lpips(outputs, sets):
    assert sets.shape[0] == outputs.shape[0]
    assert sets.shape[-3] == outputs.shape[-3]

    se = torch.mean(torch.square(outputs - sets), dim=(-3, -2, -1))
    ae = torch.mean(torch.abs(outputs - sets), dim=(-3, -2, -1))
    se_ae = 0.5 * (se + ae)
    se_ae = se_ae.squeeze(1)

    lpips_batch = lpips_loss_fn.forward(outputs.squeeze(1), sets.squeeze(1))
    lpips_table = lpips_batch.reshape(
        -1,
    )

    assert lpips_table.shape == se_ae.shape
    err = se_ae + lpips_table

    return torch.mean(err)


def rec_success_rate(rec, real, mse_threshold):
    mse_table = torch.mean(torch.square(rec - real), dim=(-3, -2, -1))
    min_mse_table, _ = torch.min(mse_table, -1)
    mse = torch.mean(min_mse_table).item()
    success_rate_mean = torch.sum(min_mse_table <= mse_threshold).item() / float(
        np.prod(list(min_mse_table.shape))
    )

    return mse, success_rate_mean


"""
	Configuration setup
"""
parser = argparse.ArgumentParser()
parser.add_argument("--data_id", type=str, help="Data name (MNIST, CIFAR10, CIFAR100).")
parser.add_argument(
    "--filename_header", type=str, help="The header for the weight data filenames."
)
parser.add_argument(
    "--seed_range", type=int, help="Number of the weight data permutation seeds."
)
parser.add_argument("--rec_name", type=str, help="Reconstructor model name.")
parser.add_argument(
    "--shadow_dir",
    type=str,
    default="./shadow_models_data",
    help="Directory containing the shadow models.",
)
parser.add_argument(
    "--stats_dir",
    type=str,
    default="./weight_stats_data",
    help="Directory containing the weight stats.",
)
parser.add_argument(
    "--reconstr_dir",
    type=str,
    default="./reconstructor_models",
    help="Directory for saving the reconstructor NN weights.",
)
parser.add_argument(
    "--config_dir",
    type=str,
    default="./config_data",
    help="Directory containing the config files.",
)
# the arguments below relevant for CelebA only
parser.add_argument(
    "--features_dir",
    type=str,
    default="./deep_features_data",
    help="Directory containing deep features.",
)
parser.add_argument(
    "--celeba_img_dir",
    type=str,
    default="./celeba_img64",
    help="Directory for saving the resized CelebA images.",
)

args = parser.parse_args()

assert args.data_id in [
    "MNIST",
    "CIFAR10",
    "CIFAR100",
    "CelebA",
], "Experiment name must be one of MNIST/CIFAR10/CIFAR100."

os.makedirs(args.reconstr_dir, exist_ok=True)

if args.data_id in ["CIFAR10", "CIFAR100"]:
    CONF = yaml.load(
        open(os.path.join(args.config_dir, "reconstruction_conf_CIFAR.yml")),
        Loader=yaml.FullLoader,
    )
else:
    CONF = yaml.load(
        open(
            os.path.join(
                args.config_dir, "reconstruction_conf_" + args.data_id + ".yml"
            )
        ),
        Loader=yaml.FullLoader,
    )

print("Data ID: ", args.data_id)
print("Training data, seed range: ", args.filename_header, args.seed_range)
print("Reconstructor name: ", args.rec_name)
print(CONF, flush=True)

"""
	Loading the test shadow models
"""
tot_seeds = args.seed_range

if args.data_id == "CIFAR10":
    train_dataset = CIFAR10(CONF['DataRoot'], download=False, train=True)
    img_data = train_dataset.data.astype(np.float32).transpose(0, 3, 1, 2) / 127.5 - 1.0
    label_data = np.array(train_dataset.targets, dtype=int)

    val_dataset = CIFAR10(CONF['DataRoot'], download=False, train=False)
    val_img_data = (
        val_dataset.data.astype(np.float32).transpose(0, 3, 1, 2) / 127.5 - 1.0
    )
    val_label_data = np.array(val_dataset.targets, dtype=int)

elif args.data_id == "CIFAR100":
    train_dataset = CIFAR100(CONF['DataRoot'], download=False, train=True)
    img_data = train_dataset.data.astype(np.float32).transpose(0, 3, 1, 2) / 127.5 - 1.0
    label_data = np.load("deep_features_data/CIFAR100_labels_train.npy").astype(int)

    val_dataset = CIFAR100(CONF['DataRoot'], download=False, train=False)
    val_img_data = (
        val_dataset.data.astype(np.float32).transpose(0, 3, 1, 2) / 127.5 - 1.0
    )
    val_label_data = np.load("deep_features_data/CIFAR100_labels_val.npy").astype(int)

elif args.data_id == "MNIST":
    transform = transforms.Compose(
        [
            transforms.Resize(32, antialias=True),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_mnist_dataset = MNIST(CONF['DataRoot'], download=False, train=True)
    img_data = transform(train_mnist_dataset.data)
    img_data = img_data.unsqueeze(1)
    label_data = train_mnist_dataset.targets

    test_mnist_dataset = MNIST(CONF['DataRoot'], download=False, train=False)
    val_img_data = transform(test_mnist_dataset.data)
    val_img_data = val_img_data.unsqueeze(1)
    val_label_data = test_mnist_dataset.targets

elif args.data_id == "CelebA":
    img_data = torch.from_numpy(
        np.load(args.celeba_img_dir + "/CelebA_img_64x64_train.npy")
    )
    label_data = torch.from_numpy(
        np.load(args.features_dir + "/CelebA_labels_train.npy")
    )

    val_img_data = torch.from_numpy(
        np.load(args.celeba_img_dir + "/CelebA_img_64x64_val.npy")
    )
    val_label_data = torch.from_numpy(
        np.load(args.features_dir + "/CelebA_labels_val.npy")
    )

cl_size = CONF["CLASS_SIZE"]

if args.data_id in ["MNIST", "CIFAR10", "CIFAR100"]:
    data_loading_fn = load_rec_data
elif args.data_id in "CelebA":
    data_loading_fn = load_celeba_data


weights_val_data, images_val_data = data_loading_fn(
    img_data=val_img_data,
    label_data=val_label_data,
    seeds=[0],
    weights_data_name=args.filename_header,
    weights_dir=args.shadow_dir,
    stats_dir=args.stats_dir,
    train=False,
    class_size=cl_size,
    group_by_class=False,
)

weights_data, images_data = data_loading_fn(
    img_data=img_data,
    label_data=label_data,
    seeds=[0],
    weights_data_name=args.filename_header,
    weights_dir=args.shadow_dir,
    stats_dir=args.stats_dir,
    train=True,
    class_size=cl_size,
    group_by_class=False,
)

print(
    "Shadow triain data shapes (single seed): ",
    tuple(weights_data.shape),
    tuple(images_data.shape),
)
print("Weights dim: ", weights_data.shape[1])


generator_input_dim = weights_data.shape[1]
out_channels = images_data.shape[-3]
res = images_data.shape[-1]
rec = Reconstructor(
    z_dim=generator_input_dim,
    gen_size=CONF["GEN_SIZE"],
    out_channels=out_channels,
    out_res=res,
).to(device)
summary(rec, input_size=(generator_input_dim,))

"""
The training loop
"""

rec.train()

cur_step = 0
generator_losses = []
generator_val_losses = []
generator_val_mses = []

# calculate validation parameters
val_batch_size = CONF["VAL_BATCH_SIZE"]
val_nb = int(len(weights_val_data) / float(val_batch_size))
# mse threshold for validation
mse_threshold = CONF["MSE_THRESHOLD"]

# determine how many seeds to load at the beginning of each epoch
if CONF["SEEDS_PER_EPOCH"] == "Auto":
    mem_per_seed = (
        8.0
        * (np.prod(tuple(images_data.shape)) + np.prod(tuple(weights_data.shape)))
        * 1e-9
    )
    n_seeds_per_epoch = min(int(CONF["DATA_MEM_LIMIT"] / mem_per_seed), tot_seeds)
    if n_seeds_per_epoch == 0:
        n_seeds_per_epoch = 1
else:
    n_seeds_per_epoch = min(CONF["SEEDS_PER_EPOCH"], tot_seeds)
steps_per_epoch = n_seeds_per_epoch * weights_data.shape[0] / CONF["BATCH_SIZE"]
checkpoint_frequency = int(5 * 1e5 / steps_per_epoch)
print(
    "n_seeds_per_epoch, checkpoint_frequency",
    n_seeds_per_epoch,
    checkpoint_frequency,
    flush=True,
)

# define the loss function
if args.data_id in ["CIFAR10", "CIFAR100"]:
    lpips_loss_fn = lpips.LPIPS(net="vgg").to(device)
    rec_loss_fn = set_softmin_mse_mae_lpips
elif args.data_id == "MNIST":
    rec_loss_fn = set_softmin_mse_mae
elif args.data_id == "CelebA":
    lpips_loss_fn = lpips.LPIPS(net="vgg").to(device)
    if cl_size == 1:
        rec_loss_fn = mse_mae_lpips
    else:
        rec_loss_fn = set_softmin_mse_mae_lpips

opt = torch.optim.Adam(rec.parameters(), lr=CONF["LR"])
cum_steps = 500
for epoch in range(CONF["TRAINING_EPOCHS"]):
    train_seeds = [
        i_seed % tot_seeds
        for i_seed in range(
            n_seeds_per_epoch * epoch, n_seeds_per_epoch * (epoch + 1), 1
        )
    ]
    weights_data, images_data = data_loading_fn(
        img_data=img_data,
        label_data=label_data,
        seeds=train_seeds,
        weights_data_name=args.filename_header,
        weights_dir=args.shadow_dir,
        stats_dir=args.stats_dir,
        train=True,
        class_size=cl_size,
        group_by_class=False,
    )

    dataset = torch_data_utils.TensorDataset(weights_data, images_data)
    dataloader = DataLoader(dataset, batch_size=CONF["BATCH_SIZE"])

    if epoch == 0:
        print(
            "Data MEM usage:\t"
            + str(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0**2)))
            + " GB",
            flush=True,
        )
        print(
            "(Weight,Img)-data shapes:\t",
            tuple(weights_data.shape),
            tuple(images_data.shape),
            flush=True,
        )

    rec.train()
    for weights, real in dataloader:
        cur_batch_size = len(weights)
        weights = weights.to(device)
        real = real.to(device)

        ### Update generator ###
        # Zero out the generator gradients
        opt.zero_grad()
        fake = torch.unsqueeze(rec(weights), 1)

        gen_loss = rec_loss_fn(fake, real)

        gen_loss.backward()
        opt.step()

        # Keep track of the generator losses
        generator_losses += [gen_loss.item()]
        #

        cur_step += 1

    # Validation loss
    rec.eval()
    reconstructions = []
    with torch.no_grad():
        for b in range(val_nb + 1):
            weights_val_batch = weights_val_data[
                b
                * val_batch_size : min((b + 1) * val_batch_size, len(weights_val_data))
            ]
            weights_val_batch = weights_val_batch.to(device)
            reconstructions.append(rec(weights_val_batch))
    reconstructions = torch.cat(reconstructions, 0).cpu()
    reconstructions = reconstructions.unsqueeze(1)
    gen_val_mse, success_rate_mean = rec_success_rate(
        reconstructions, images_val_data, mse_threshold
    )
    gen_mean = sum(generator_losses[-cum_steps:]) / len(generator_losses[-cum_steps:])
    print(
        f"Epoch {epoch}, step {cur_step}: Training loss: {gen_mean}, Val MSE loss: {gen_val_mse}"
    )
    print("Reconstruction success rate:", success_rate_mean, flush=True)

    if epoch % checkpoint_frequency == checkpoint_frequency - 1:
        torch.save(
            rec.state_dict(),
            args.reconstr_dir
            + "/"
            + args.rec_name
            + "_trained_"
            + str(epoch + 1)
            + "EP.pth",
        )

torch.save(
    rec.state_dict(),
    args.reconstr_dir + "/" + args.rec_name + "_trained_" + str(epoch + 1) + "EP.pth",
)
