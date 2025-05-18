import numpy as np
import torch
import random
import itertools as it


def permute_data(data, seed):
    rng = np.random.default_rng(seed=seed)
    permuted_inds = rng.permutation(data.shape[0]).astype(int)

    return np.take(data, permuted_inds, axis=0)


def get_balanced_sets(data, labels, class_size, seed, group_by_class=False):
    n_classes = len(set(labels.flatten().tolist()))
    assert set(labels.flatten().tolist()) == set([i for i in range(n_classes)])
    data_shape = data.shape[1:]

    data = np.asarray(data)
    labels = np.asarray(labels)

    data_perm = permute_data(data, seed)
    labels_perm = permute_data(labels, seed).reshape(
        -1,
    )

    class_inds = [
        (labels_perm == cl)
        .nonzero()[0]
        .reshape(
            -1,
        )
        for cl in range(n_classes)
    ]
    n_sets = min([int(len(inds) / float(class_size)) for inds in class_inds])
    class_inds = [inds[: class_size * n_sets] for inds in class_inds]

    img_sets = [
        torch.from_numpy(np.take(data_perm, inds, axis=0))
        .float()
        .reshape(
            (
                n_sets,
                class_size,
            )
            + data_shape
        )
        for inds in class_inds
    ]
    img_sets = torch.cat(img_sets, 1)
    label_sets = torch.cat(
        [
            torch.ones(size=(n_sets, class_size, 1), dtype=int) * cl
            for cl in range(n_classes)
        ],
        1,
    )

    if group_by_class:
        img_sets = img_sets.reshape(
            (
                len(img_sets),
                n_classes,
                class_size,
            )
            + data_shape
        )

    return img_sets, label_sets


def load_rec_data(
    img_data,
    label_data,
    seeds,
    weights_data_name,
    weights_dir,
    stats_dir,
    train=True,
    class_size=3,
    group_by_class=True,
):

    weights_data = []
    images_data = []
    w_mean = np.load(stats_dir + "/" + "stats_mean_" + weights_data_name + ".npy")
    w_std = np.load(stats_dir + "/" + "stats_std_" + weights_data_name + ".npy")
    for seed in seeds:
        # print("Loading training data seed ", seed, flush=True)
        if train:
            train_weights_seed = np.load(
                weights_dir
                + "/"
                + weights_data_name
                + "_train_seed_"
                + str(seed)
                + ".npy"
            )
        else:
            train_weights_seed = np.load(
                weights_dir
                + "/"
                + weights_data_name
                + "_test_seed_"
                + str(seed)
                + ".npy"
            )
        training_sets, _ = get_balanced_sets(
            img_data,
            label_data,
            class_size=class_size,
            seed=seed,
            group_by_class=group_by_class,
        )
        images_data.append(training_sets[: len(train_weights_seed)])
        weights_data.append(train_weights_seed)

    images_data = torch.cat(images_data, 0)

    weights_data = np.concatenate(weights_data, 0)
    assert np.sum(np.isnan(w_std)) == 0, "NAN values in w_std!"

    # handling the possible vanishing variance
    if np.sum(w_std == 0.0) > 0:
        nonzero_inds = np.nonzero(w_std)[1]
        w_std = np.take(w_std, nonzero_inds, axis=1)
        w_mean = np.take(w_mean, nonzero_inds, axis=1)
        weights_data = np.take(weights_data, nonzero_inds, axis=1)
    weights_data = (weights_data - w_mean) / w_std
    assert (
        np.sum(np.isnan(weights_data)) == 0
    ), "NAN values in weights_data after normalisation!"
    weights_data = torch.from_numpy(weights_data.astype(np.float32))

    return weights_data, images_data


def load_celeba_data(
    img_data,
    label_data,
    class_size,
    seeds,
    weights_data_name,
    weights_dir,
    stats_dir,
    train=True,
    group_by_class=False,
):

    assert len(img_data) == len(label_data)
    w_mean = np.load(stats_dir + "/" + "stats_mean_" + weights_data_name + ".npy")
    w_std = np.load(stats_dir + "/" + "stats_std_" + weights_data_name + ".npy")
    weights_data = []
    images_data = []
    for seed in seeds:
        if train:
            weights_seed = np.load(
                weights_dir
                + "/"
                + weights_data_name
                + "_train_seed_"
                + str(seed)
                + ".npy"
            )
        else:
            weights_seed = np.load(
                weights_dir
                + "/"
                + weights_data_name
                + "_test_seed_"
                + str(seed)
                + ".npy"
            )
        rng = np.random.default_rng(seed=seed)
        perm_inds = torch.from_numpy(
            rng.permutation(range(len(img_data))).reshape(-1, 1, 1, 1)
        )
        img_data_perm = torch.take_along_dim(img_data, perm_inds, dim=0)
        labels_perm = torch.take_along_dim(
            label_data,
            perm_inds.reshape(
                -1,
            ),
            dim=0,
        )
        n_sets_per_seed = len(weights_seed)
        img_sets = []
        i = 0
        while len(img_sets) < n_sets_per_seed:
            l = labels_perm[i]
            mask = labels_perm == l
            if torch.sum(mask) > class_size:
                set_i = img_data_perm[mask]
                img_sets.append(torch.unsqueeze(set_i[:class_size], 0))
            i += 1
        img_sets = torch.cat(img_sets, 0)

        images_data.append(img_sets)
        weights_data.append(weights_seed)

    images_data = torch.cat(images_data, 0)

    weights_data = np.concatenate(weights_data, 0)
    assert np.sum(np.isnan(w_std)) == 0, "NAN values in w_std!"
    if np.sum(w_std == 0.0) > 0:
        # print('Found weights with zero variance!', flush=True)
        nonzero_inds = np.nonzero(w_std)[1]
        w_std = np.take(w_std, nonzero_inds, axis=1)
        w_mean = np.take(w_mean, nonzero_inds, axis=1)
        weights_data = np.take(weights_data, nonzero_inds, axis=1)
    weights_data = (weights_data - w_mean) / w_std
    assert (
        np.sum(np.isnan(weights_data)) == 0
    ), "NAN values in weights_data after normalisation!"
    weights_data = torch.from_numpy(weights_data.astype(np.float32))

    return weights_data, images_data
