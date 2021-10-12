from os.path import dirname, join, realpath

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    fname = join(dirname(__file__), "convergence_data.txt")
    data = np.genfromtxt(
        fname, delimiter=",", missing_values=["nan"], filling_values=np.nan
    )
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
