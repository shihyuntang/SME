# -*- coding: utf-8 -*-
import re
from os.path import dirname, join, realpath

import matplotlib.pyplot as plt
import numpy as np


def get_parameters_from_logfile_name(logfile):
    match = re.search(r"logg\/([\d.]+)\/monh\/([+-.\d]+)\/teff\/([\d.]+)\/", logfile)
    logg = float(match.group(1))
    monh = float(match.group(2))
    teff = float(match.group(3))
    return teff, logg, monh


def get_parameters_from_log(logfile):
    with open(logfile) as f:
        sme_log = f.read()

    match = re.search(r"teff\s*([\d.]+) \+- ([\d.]+)", sme_log)
    teff = float(match.group(1))
    teff_unc = float(match.group(2))

    match = re.search(r"logg\s*([\d.]+) \+- ([\d.]+)", sme_log)
    logg = float(match.group(1))
    logg_unc = float(match.group(2))

    match = re.search(r"monh\s*([\d.]+) \+- ([\d.]+)", sme_log)
    monh = float(match.group(1))
    monh_unc = float(match.group(2))

    return teff, logg, monh


if __name__ == "__main__":
    fname = join(dirname(__file__), "convergence_data.txt")
    with open(fname) as f:
        logfiles = f.readlines()

    data = np.zeros((len(logfiles), 6))
    for i, logfile in enumerate(logfiles):
        logfile = join(dirname(fname), logfile[:-1])
        data[i, :3] = get_parameters_from_logfile_name(logfile)
        data[i, 3:] = get_parameters_from_log(logfile)

    teffs = data[:, 0]
    loggs = data[:, 1]
    monhs = data[:, 2]
    result_teff = data[:, 3]
    result_logg = data[:, 4]
    result_monh = data[:, 5]

    unique_teff = np.sort(np.unique(teffs))
    unique_logg = np.sort(np.unique(loggs))
    unique_monh = np.sort(np.unique(monhs))

    # h, *_ = plt.hist(result_teff.ravel(), bins="auto")
    # plt.vlines(5770, 0, h.max(), color="k")
    # plt.xlabel("Teff [K]")
    # plt.show()

    # Distribution of new teffs, depending on old teff
    plt.plot(result_teff, teffs, "k+", alpha=0.5)
    plt.xlabel("Final Teff [K]")
    plt.ylabel("Initial Teff [K]")
    plt.show()

    # Distribution of new values
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(teffs.ravel(), loggs.ravel(), monhs.ravel())
    ax.scatter(result_teff.ravel(), result_logg.ravel(), result_monh.ravel())
    plt.show()
    pass
