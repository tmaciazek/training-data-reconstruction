import torch
import resource
import lpips
import sys
import argparse
import yaml
import sys, os

import torch.utils.data as data_utils

from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, CelebA
from torchvision.io import read_image
from torchvision.utils import make_grid

from torch.distributions.multivariate_normal import MultivariateNormal

import matplotlib.pyplot as plt

from data_utils import *
from models import Reconstructor

print("CUDA available?\t", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def min_mse_mae(outputs, sets):
    assert sets.shape[0] == outputs.shape[0]
    assert sets.shape[-3] == outputs.shape[-3]

    B = sets.shape[0]
    CL = sets.shape[1]
    CH = sets.shape[-3]
    RES = sets.shape[-1]

    se = torch.mean(torch.square(outputs - sets), dim=(-3, -2, -1))
    ae = torch.mean(torch.abs(outputs - sets), dim=(-3, -2, -1))
    err = 0.5 * (se + ae)

    return torch.min(err, dim=-1)


def min_mse_mae_lpips(outputs, sets):
    assert sets.shape[0] == outputs.shape[0]
    assert sets.shape[-3] == outputs.shape[-3]

    B = sets.shape[0]
    N_CL = sets.shape[1]
    CL_SIZE = sets.shape[2]
    CH = sets.shape[-3]
    RES = sets.shape[-1]

    se = torch.mean(torch.square(outputs - sets), dim=(-3, -2, -1))
    ae = torch.mean(torch.abs(outputs - sets), dim=(-3, -2, -1))
    se_ae = 0.5 * (se + ae)
    out_repeat = outputs.repeat_interleave(CL_SIZE, dim=2)
    out_repeat = out_repeat.reshape(-1, CH, RES, RES)
    lpips_batch = lpips_loss_fn.forward(out_repeat, sets.reshape(-1, CH, RES, RES))
    lpips_table = lpips_batch.reshape(B, N_CL, CL_SIZE)
    assert lpips_table.shape == se_ae.shape
    err = se_ae + lpips_table

    return torch.min(err, dim=-1)


"""
	Configuration setup
"""
parser = argparse.ArgumentParser()
parser.add_argument("--data_id", type=str, help="Data name (MNIST, CIFAR10, CIFAR100).")
parser.add_argument(
    "--train_filename",
    type=str,
    help="The header of the weight data filenames (training shadow models).",
)
parser.add_argument(
    "--val_filename",
    type=str,
    help="The header of the weight data filenames (validation shadow models).",
)
parser.add_argument(
    "--random", type=str, default="F", help="Generate images from random weights?"
)
parser.add_argument(
    "--test_seed_range",
    type=int,
    default=1,
    help="Number of the validation weight data permutation seeds.",
)
parser.add_argument("--rec_name", type=str, help="Reconstructor model name.")
parser.add_argument(
    "--conditional", type=str, default="T", help="Using a conditional reconstructor NN?"
)
parser.add_argument(
    "--filename_appendix",
    type=str,
    default="min_mse_table",
    help="min-MSE table file name appendix",
)
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
    help="Directory containing the reconstructor NN weights.",
)
parser.add_argument(
    "--minmse_dir",
    type=str,
    default="./min_mse_data",
    help="Directory saving the min-MSE tables.",
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
assert args.conditional in ["F", "T"], "--conditional must be one of T/F"
assert args.random in ["F", "T"], "--random must be one of T/F"

flag_cond = args.conditional == "T"

os.makedirs(args.minmse_dir, exist_ok=True)

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
print(
    "Training data, test data, test seed range: ",
    args.train_filename,
    args.val_filename,
    args.test_seed_range,
)
print("Reconstructor name: ", args.rec_name)
print(CONF, flush=True)


"""
	Constructing the test data
"""

if args.data_id == "CIFAR10":
    val_dataset = CIFAR10(CONF["DataRoot"], download=False, train=False)
    val_img_data = (
        val_dataset.data.astype(np.float32).transpose(0, 3, 1, 2) / 127.5 - 1.0
    )
    val_label_data = np.array(val_dataset.targets, dtype=int)

elif args.data_id == "CIFAR100":
    val_dataset = CIFAR100(CONF["DataRoot"], download=False, train=False)
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

    test_mnist_dataset = MNIST(CONF["DataRoot"], download=False, train=False)
    val_img_data = transform(test_mnist_dataset.data)
    val_img_data = val_img_data.unsqueeze(1)
    val_label_data = test_mnist_dataset.targets

elif args.data_id == "CelebA":
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
    seeds=[i for i in range(args.test_seed_range)],
    weights_data_name=args.val_filename,
    weights_dir=args.shadow_dir,
    stats_dir=args.stats_dir,
    train=False,
    class_size=cl_size,
    group_by_class=flag_cond,
)

print(
    "Shadow validation data shapes (single seed): ",
    tuple(weights_val_data.shape),
    tuple(images_val_data.shape),
)
print("Weights dim: ", weights_val_data.shape[1])

"""
	Resconstructor setup
"""

if flag_cond and (args.data_id in ["MNIST", "CIFAR10", "CIFAR100"]):
    generator_input_dim = weights_val_data.shape[1] + 10
else:
    generator_input_dim = weights_val_data.shape[1]
out_channels = images_val_data.shape[-3]
res = images_val_data.shape[-1]
rec = Reconstructor(
    z_dim=generator_input_dim,
    gen_size=CONF["GEN_SIZE"],
    out_channels=out_channels,
    out_res=res,
).to(device)

rec.load_state_dict(
    torch.load(args.reconstr_dir + "/" + args.rec_name + ".pth", map_location=device)
)
print("Loaded weights from ", args.rec_name)

print(
    f"The Reconstructor NN has {sum(p.numel() for p in rec.parameters()):,} parameters"
)

"""
	Sample min-MSE from generative and from true inputs
"""

w_cov = torch.from_numpy(
    np.load(args.stats_dir + "/" + "stats_covnorm_" + args.train_filename + ".npy")
)
w_cov = w_cov.to(device).to(torch.float32)
# add jitter term to avoid having singular covariance matrix
jitter = 1e-5
w_cov = w_cov + jitter * torch.eye(w_cov.shape[0]).to(device)

if args.random == "T":
    print("Sampling random weights...", flush=True)
    rand_mean = torch.zeros(weights_val_data.shape[1]).to(device).to(torch.float32)
    mvn = MultivariateNormal(loc=rand_mean, covariance_matrix=w_cov)
    weights_val_data = mvn.sample([weights_val_data.shape[0]]).cpu()


def append_one_hot_labels(weights_batch):
    one_hot_labels = torch.eye(10)
    one_hot_labels = one_hot_labels.repeat(len(weights_batch), 1)
    weights_batch = weights_batch.repeat_interleave(repeats=10, dim=0)
    weights_batch = torch.cat([weights_batch, one_hot_labels], 1)

    return weights_batch


val_batch_size = 10
val_nb = int(len(weights_val_data) / float(val_batch_size))


if args.data_id != "MNIST":
    lpips_loss_fn = lpips.LPIPS(net="vgg").to(device)

rec.eval()

reconstructions = []
mse_threshold = CONF["MSE_THRESHOLD"]
with torch.no_grad():
    for b in range(val_nb + 1):
        weights_val_batch = weights_val_data[
            b * val_batch_size : min((b + 1) * val_batch_size, len(weights_val_data))
        ]
        img_val_batch = images_val_data[
            b * val_batch_size : min((b + 1) * val_batch_size, len(weights_val_data))
        ]
        assert len(weights_val_batch) == len(img_val_batch)

        if flag_cond and (args.data_id in ["MNIST", "CIFAR10", "CIFAR100"]):
            weights_val_batch = append_one_hot_labels(weights_val_batch)
        weights_val_batch = weights_val_batch.to(device)
        img_val_batch = img_val_batch.to(device)
        if args.data_id == "CelebA":
            img_val_batch = img_val_batch.unsqueeze(1)
        B = len(img_val_batch)
        CH, H, W = img_val_batch.shape[-3:]
        N_CL = img_val_batch.shape[1]
        rec_batch = rec(weights_val_batch)
        rec_batch = rec_batch.reshape(B, N_CL, 1, CH, H, W)
        if args.data_id == "MNIST":
            _, min_inds_batch = min_mse_mae(rec_batch, img_val_batch)
        else:
            _, min_inds_batch = min_mse_mae_lpips(rec_batch, img_val_batch)

        # matched_img = torch.cat([img_set[ind].unsqueeze(0) for img_set, ind in zip(img_val_batch,min_inds)],0)
        matched_img = []
        for orig_set, min_inds in zip(img_val_batch, min_inds_batch):
            matches = [
                torch.cat(
                    [
                        img_set[ind].unsqueeze(0)
                        for img_set, ind in zip(orig_set, min_inds)
                    ],
                    0,
                )
            ]
            matched_img.append(torch.unsqueeze(torch.cat(matches, 0), 0))
        matched_img = torch.cat(matched_img, 0)
        if args.data_id == "MNIST":
            matched_img = torch.cat([matched_img, matched_img, matched_img], -3)
            rec_batch = torch.cat([rec_batch, rec_batch, rec_batch], -3)
        rec_batch = rec_batch.squeeze(2)
        se_table = torch.mean(torch.square(matched_img - rec_batch), dim=(-3, -2, -1))
        print(se_table.shape)
        positive = [
            np.where((mses.cpu().numpy() <= mse_threshold))[0] for mses in se_table
        ]
        matched_img = (matched_img + 1) * 127.5
        rec_batch = (rec_batch + 1) * 127.5
        if args.data_id == "CelebA":
            bsize = len(rec_batch)
            rec_batch = rec_batch.squeeze(1)
            matched_img = matched_img.squeeze(1)
            positive = [len(pos) > 0 for pos in positive]
            print(positive)

            tick = read_image("figures/tick64.jpg").to(float).unsqueeze(0)
            xmark = read_image("figures/cross64.jpg").to(float).unsqueeze(0)

            to_plot = torch.cat([matched_img, rec_batch], 0)
            ticks = torch.cat([tick if pos else xmark for pos in positive], 0)
            to_plot = torch.cat([to_plot, ticks], 0)
            image_grid = make_grid(to_plot, nrow=bsize, pad_value=255.0)
            fig = plt.gcf()
            fig.set_size_inches(8.3, 3.32)
            plt.imshow(image_grid.numpy().transpose(1, 2, 0).astype("uint8"))
            plt.axis("off")
            plt.show()

        else:
            tick = read_image("figures/tick32.jpg").to(float).unsqueeze(0)
            xmark = read_image("figures/cross32.jpg").to(float).unsqueeze(0)

            for orig_img, rec_img, pos in zip(matched_img, rec_batch, positive):
                print(orig_img.shape, pos)

                to_plot = torch.cat([orig_img, rec_img], 0)
                ticks = torch.cat([tick if i in pos else xmark for i in range(10)], 0)
                to_plot = torch.cat([to_plot, ticks], 0)
                image_grid = make_grid(to_plot, nrow=10, pad_value=255.0)
                fig = plt.gcf()
                fig.set_size_inches(8.3, 3.32)
                plt.imshow(image_grid.numpy().transpose(1, 2, 0).astype("uint8"))
                plt.axis("off")
                plt.show()
