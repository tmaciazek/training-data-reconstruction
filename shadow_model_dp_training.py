import torch
import resource
import time
import argparse
import yaml
import sys, os

import torch.utils.data as data_utils

from torch import nn
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAccuracy

import matplotlib.pyplot as plt

from torchsummary import summary

from opacus import PrivacyEngine, accountants

from data_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_id", type=str, help="Dtaset name (MNIST, CIFAR10).")
parser.add_argument(
    "--split", type=str, default="train", help="Generate train/test shadow models?"
)
parser.add_argument(
    "--permutation_seed", type=int, help="Random seed used to permute the data."
)
parser.add_argument(
    "--models_per_seed",
    type=int,
    help="How many shadow models per seed? See documentation.",
)
parser.add_argument(
    "--config_dir",
    type=str,
    default="./config_data",
    help="Directory containing the config files.",
)
parser.add_argument(
    "--features_dir",
    type=str,
    default="./deep_features_data",
    help="Directory containing the deep features.",
)
parser.add_argument(
    "--shadow_dir",
    type=str,
    default="./shadow_models_data",
    help="Directory for saving the shadow models.",
)
parser.add_argument(
    "--filename_appendix",
    type=str,
    default="classifier",
    help="Shadow model file name appendix",
)

args = parser.parse_args()
assert args.data_id in ["MNIST", "CIFAR10"], "data_id must be one of MNIST, CIFAR10"
assert args.split in ["test", "train"], "split must be one of train/test"

os.makedirs(args.shadow_dir, exist_ok=True)


if args.data_id == "CIFAR10":
    CONF = yaml.load(
        open(os.path.join(args.config_dir, "shadow_conf_multiclass_CIFAR.yml")),
        Loader=yaml.FullLoader,
    )
else:
    CONF = yaml.load(
        open(
            os.path.join(
                args.config_dir, "shadow_conf_multiclass_" + args.data_id + ".yml"
            )
        ),
        Loader=yaml.FullLoader,
    )

assert (CONF["TRAINING_EPOCHS"] == "Auto") or type(
    CONF["TRAINING_EPOCHS"]
) == int, "TRAINING_EPOCHS must be Auto or an integer"
assert (CONF["DP_DELTA"] == "Auto") or type(
    CONF["DP_DELTA"]
) == int, "DP_DELTA must be Auto or an integer"
assert (CONF["BATCH_SIZE"] in ["Max"]) or type(
    CONF["BATCH_SIZE"]
) == int, "BATCH_SIZE must be Max or an integer"
assert CONF["OPTIMIZER"] in ["SGD", "Adam"], "OPTIMIZER must be one of SGD/Adam"

print("Data ID: ", args.data_id)
print("Shadow model split: ", args.split)

print(CONF, flush=True)

"""
	Load features data
"""

features_train = torch.from_numpy(
    np.load(args.features_dir + "/" + args.data_id + "_features_train.npy")
)
labels_train = torch.from_numpy(
    np.load(args.features_dir + "/" + args.data_id + "_labels_train.npy")
)
features_val = torch.from_numpy(
    np.load(args.features_dir + "/" + args.data_id + "_features_val.npy")
)
labels_val = torch.from_numpy(
    np.load(args.features_dir + "/" + args.data_id + "_labels_val.npy")
)

"""
	Prepare the head NN
"""

features_dim = features_train.shape[1]

head = nn.Sequential(nn.Linear(in_features=features_dim, out_features=10, bias=True))
print(summary(head, (features_dim,)))

"""
	Classifier pre-training
"""

class_size = CONF["CLASS_SIZE"]
seed = args.permutation_seed
models_per_seed = args.models_per_seed
if CONF["TRAINING_EPOCHS"] == "Auto":
    if args.data_id == "MNIST":
        n_epochs = 26 + 6 * class_size
    elif args.data_id in ["CIFAR10", "CIFAR100"]:
        n_epochs = 38 + 10 * class_size
    print("Training epochs no.:\t", n_epochs, flush=True)
else:
    n_epochs = CONF["TRAINING_EPOCHS"]
lr = CONF["LR"]

if CONF["BATCH_SIZE"] == "Full":
    batch_size = class_size * 10
elif CONF["BATCH_SIZE"] == "Max":
    batch_size = class_size * 10 - 1
else:
    batch_size = CONF["BATCH_SIZE"]
init_std = CONF["INIT_STD"]


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, init_std)  # 0.002
        nn.init.constant_(m.bias, 0)


# prepare shadow model training sets for CIFAR/MNIST
if args.split == "train":
    training_sets, label_sets = get_balanced_sets(
        features_train, labels_train, class_size=class_size, seed=seed
    )
else:
    training_sets, label_sets = get_balanced_sets(
        features_val, labels_val, class_size=class_size, seed=seed
    )
print("Total no. of shadow models available:\t", len(training_sets), flush=True)
assert (
    len(training_sets) >= models_per_seed
), "models_per_seed must be at most the total no. of shadow models available"

# define the loss function
loss_fn = nn.CrossEntropyLoss()

opt_name = CONF["OPTIMIZER"]

"""
	Shadow head training loop
"""

trained_models_weights = []
tot_time = 0.0
running_acc = []
for training_no in range(models_per_seed):
    print(
        "Seed, training_no:\t"
        + str(seed)
        + "\t"
        + str(training_no)
        + "/"
        + str(models_per_seed),
        flush=True,
    )
    tic = time.perf_counter()
    head = head.apply(weights_init)
    if opt_name == "SGD":
        opt = torch.optim.SGD(
            head.parameters(), lr=lr, weight_decay=CONF["WEIGHT_DECAY"]
        )
    else:
        opt = torch.optim.Adam(
            head.parameters(), lr=lr, weight_decay=CONF["WEIGHT_DECAY"]
        )

    X_train = training_sets[training_no]
    labels = label_sets[training_no]

    dataset = data_utils.TensorDataset(X_train, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    privacy_engine = PrivacyEngine(accountant="prv")
    head, opt, dataloader = privacy_engine.make_private(
        module=head,
        optimizer=opt,
        data_loader=dataloader,
        noise_multiplier=CONF["DP_NOISE"],
        max_grad_norm=CONF["DP_CLIPNORM"],
    )

    for epoch in range(n_epochs):
        for X, label in dataloader:
            opt.zero_grad()
            outputs = head(X)
            loss = loss_fn(
                outputs,
                label.reshape(
                    -1,
                ),
            )
            loss.backward()
            opt.step()

    with torch.no_grad():
        val_outputs = head(features_val)
        _, predicted = torch.max(val_outputs.data, 1)
        acc = 100 * (predicted == labels_val).sum().item() / float(len(val_outputs))
        print("Final training loss, final val acc:", loss.item(), acc)
        running_acc.append(acc)

    weights_flat = torch.cat([p.data.flatten() for p in head.parameters()], 0).numpy()
    trained_models_weights.append(weights_flat)

    toc = time.perf_counter()
    tot_time += toc - tic


app = args.filename_appendix + "_DP"

if args.split == "train":
    filename = (
        args.data_id
        + "_"
        + app
        + "_"
        + "N"
        + str(10 * class_size)
        + "_train_seed_"
        + str(seed)
    )
else:
    filename = (
        args.data_id
        + "_"
        + app
        + "_"
        + "N"
        + str(10 * class_size)
        + "_test_seed_"
        + str(seed)
    )

np.save(args.shadow_dir + "/" + filename + ".npy", np.array(trained_models_weights))


print("Training time per shadow model [s]:\t", tot_time / models_per_seed)
print("Total training time [min]:\t", tot_time / 60)
print("Mean, std of val accuracy:\t", np.mean(running_acc), np.std(running_acc))

if CONF["DP_DELTA"] == "Auto":
    delta = 1.0 / np.power(float(class_size * 10), 1.1)
else:
    delta = CONF["DP_DELTA"]
print("Delta:", delta)
epsilon = privacy_engine.accountant.get_epsilon(delta=delta)
print("Final epsilon:", epsilon)
