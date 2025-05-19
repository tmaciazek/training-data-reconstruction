import torch
import resource
import lpips
import sys
import os.path
import numpy as np
import argparse
import sys, os

from scipy.stats import norm
from scipy.special import erf, erfinv


import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument(
    "--min_mse_table",
    type=str,
    help="The name of the files containing the min-MSE tables for H0 and H1.",
)
parser.add_argument(
    "--class_ind",
    type=int,
    default=-1,
    help="The index of the class for which to build the ROC (relevant only for the conditional reconstructor).",
)
parser.add_argument(
    "--mse_threshold",
    type=float,
    help="The nearest-neighbour MSE threshold.",
)
parser.add_argument(
    "--minmse_dir",
    type=str,
    default="./min_mse_data",
    help="Directory containing the min-MSE tables.",
)

parser.add_argument(
    "--logscale",
    type=str,
    default="F",
    help="Plot the ROC curves in logscale?",
)
args = parser.parse_args()

print("Using the min-MSE file: ", args.min_mse_table, flush=True)


def transf(x):
    return np.log(x / 4.0) - np.log(1.0 - x / 4.0)


def tpr_fun(fpr, c0, s0, c1, s1):
    return (
        1.0 - erf((c0 - c1) / (np.sqrt(2.0) * s0) + s1 * erfinv(1.0 - 2.0 * fpr) / s0)
    ) / 2.0


min_mse_table_H0 = np.load(args.minmse_dir + "/" + args.min_mse_table + "_H0.npy")
min_mse_table_H1 = np.load(args.minmse_dir + "/" + args.min_mse_table + "_H1.npy")

tau = args.mse_threshold
tpr_nn = np.sum(min_mse_table_H0 <= tau) / float(np.prod(list(min_mse_table_H0.shape)))
fpr_nn = np.sum(min_mse_table_H1 <= tau) / float(np.prod(list(min_mse_table_H1.shape)))
print("TPR, FPR corresponding to the nearest-neighbour MSE threshold: ", tpr_nn, fpr_nn)

min_mse_table_H0 = min_mse_table_H0[:, args.class_ind]
min_mse_table_H1 = min_mse_table_H1[:, args.class_ind]

var1 = transf(min_mse_table_H1)
var0 = transf(min_mse_table_H0)

(mu0, sigma0) = norm.fit(var0)
(mu1, sigma1) = norm.fit(var1)
print("Fitted gaussians (mean, std) for H0, H1: ", (mu0, sigma0), (mu1, sigma1))

fig, ax = plt.subplots()

bins = "auto"
n0, bins0, _ = ax.hist(var0, bins=bins, alpha=0.5, density=True, label=r"$H_0$")
y0 = norm.pdf(bins0, mu0, sigma0)
plt.plot(bins0, y0, "b--")
n1, bins1, _ = ax.hist(var1, bins=bins, alpha=0.5, density=True, label=r"$H_1$")
y1 = norm.pdf(bins1, mu1, sigma1)
plt.plot(bins1, y1, color="orange", linestyle="--")
ax.set_xlabel(r"$\phi$", fontsize=15)
ax.set_ylabel(r"$counts\ (normalized)$", fontsize=15)
ax.legend(fontsize=15)
plt.savefig("min_MSE_histogram.pdf")
plt.clf()


tau_list = np.linspace(0.0, 1.0, 2000)
tpr_list = []
fpr_list = []
# calculate the cumulative ROC
for tau in tau_list:
    tpr = np.sum(min_mse_table_H0 <= tau) / float(np.prod(list(min_mse_table_H0.shape)))
    tpr_list += [tpr]
    fpr = np.sum(min_mse_table_H1 <= tau) / float(np.prod(list(min_mse_table_H1.shape)))
    fpr_list += [fpr]
tpr_list = np.array(tpr_list)
fpr_list = np.array(fpr_list)

# log-likelihood ratios for the Neyman-Pearson criterion
lr_tp = norm.logpdf(var0, loc=mu0, scale=sigma0) - norm.logpdf(
    var0, loc=mu1, scale=sigma1
)
lr_fp = norm.logpdf(var1, loc=mu0, scale=sigma0) - norm.logpdf(
    var1, loc=mu1, scale=sigma1
)

C_mesh = np.linspace(-8, 8, 2000)
N_tp = var0.shape[0]
tpr_NP0 = []
fpr_NP0 = []
for C in C_mesh:
    tpr_C0 = np.sum(lr_tp > C) / var0.shape[0]
    fpr_C0 = np.sum(lr_fp > C) / var1.shape[0]
    tpr_NP0.append(tpr_C0)
    fpr_NP0.append(fpr_C0)
tpr_NP0 = np.array(tpr_NP0)
fpr_NP0 = np.array(fpr_NP0)


"""
	Plotting
"""
fig, ax = plt.subplots()

logscale = args.logscale == "T"

mask = (tpr_list > 1e-5) & (fpr_list > 1e-5)
tpr_list = tpr_list[mask]
fpr_list = fpr_list[mask]

mask1 = fpr_NP0 > 1e-5
tpr_NP0 = tpr_NP0[mask1]
fpr_NP0 = fpr_NP0[mask1]
if not logscale:
    tpr_list = np.append([0.0], tpr_list)
    fpr_list = np.append([0.0], fpr_list)
    fpr_NP0 = np.append(fpr_NP0, [0.0])
    tpr_NP0 = np.append(tpr_NP0, [0.0])
tpr_list = np.append(tpr_list, [1.0])
fpr_list = np.append(fpr_list, [1.0])

ax.plot(fpr_NP0, tpr_NP0, label=r"N-P ROC $H_0\ vs.\ H_1$")
ax.plot(fpr_list, tpr_list, linestyle="--", label=r"Cumulative ROC $H_0\ vs.\ H_1$")
ax.plot(
    fpr_NP0,
    tpr_fun(fpr_NP0, mu0, sigma0, mu1, sigma1),
    linestyle="--",
    label=r"Analytic ROC $H_0\ vs.\ H_1$",
)


if logscale:
    ax.plot([1e-4, 1.0], [1e-4, 1.0], color="black", linestyle="--", label="FPR=TPR")
    ax.set_xlim([1e-4, 1.0])
    ax.set_xscale("log")
    ax.set_yscale("log")
else:
    ax.plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--", label="FPR=TPR")

ax.set_xlabel("False Positive Rate", fontsize=15)
ax.set_ylabel("True Positive Rate", fontsize=15)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.95, box.height * 0.95])
fig.legend(loc=(box.width * 0.5, box.y0 * 1.5), ncol=1, fontsize=12, shadow=True)

if logscale:
    ax.set_title("Analytic vs. sampled ROC, log-scale", fontsize=15)
else:
    ax.set_title("Analytic vs. sampled ROC, linear scale", fontsize=15)

plt.savefig("ROC_curve.pdf")
