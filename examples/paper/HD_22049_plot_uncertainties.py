""" Minimum working example of an SME script 
"""
import os
import os.path
import re
from os.path import dirname, join, realpath
import datetime

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import coordinates as coord
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from data_sources.StellarDB import StellarDB
from pysme import sme as SME
from pysme import util
from pysme.abund import Abund
from pysme.gui import plot_plotly
from pysme.iliffe_vector import Iliffe_vector
from pysme.linelist.vald import ValdFile
from pysme.persistence import save_as_idl
from pysme.solve import solve
from pysme.synthesize import synthesize_spectrum
from pysme.continuum_and_radial_velocity import determine_radial_velocity
from scipy.linalg import lstsq, solve_banded
from scipy.ndimage.filters import gaussian_filter1d, median_filter
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from scipy.optimize.minpack import curve_fit
from scipy.stats import gennorm, norm, skewnorm, exponnorm
from scipy.special import owens_t
from scipy.integrate import quad
from tqdm import tqdm


def find_roots(x, y):
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    return x[:-1][s] + np.diff(x)[s] / (np.abs(y[1:][s] / y[:-1][s]) + 1)


def cdf(x, mu, alpha, *beta):
    """
    Cumulative distribution function of the generalized normal distribution
    the factor sqrt(2) is a conversion between generalized and regular normal distribution
    """
    # return gennorm.cdf(x, beta, loc=mu, scale=alpha * np.sqrt(2))
    return norm.cdf(x, loc=mu, scale=alpha)
    # return skewnorm.cdf(x, beta, loc=mu, scale=alpha)
    # return cauchy.cdf(x, loc=mu, scale=alpha)
    # This is for the skewed normal distribution but faster
    # return norm.cdf(x, loc=mu, scale=alpha) - 2 * owens_t((x - mu) / alpha, skew)
    # return exponnorm.cdf(x, beta, loc=mu, scale=alpha)


def std(mu, alpha, *beta):
    """ 1 sigma (68.27 %) quantile, assuming symmetric distribution """
    # interval = gennorm.interval(0.6827, beta, loc=mu, scale=alpha * np.sqrt(2))
    # interval = exponnorm.interval(0.6827, beta, loc=mu, scale=alpha)
    interval = norm.interval(0.6827, loc=mu, scale=alpha)
    # interval = cauchy.interval(0.6827, loc=mu, scale=alpha)
    # interval = skewnorm.interval(0.6827, skew, loc=mu, scale=alpha)
    sigma = (interval[1] - interval[0]) / 2
    return sigma, interval


def pdf(x, mu, alpha, *beta):
    return norm.pdf(x, loc=mu, scale=alpha)
    # return gennorm.pdf(x, beta, loc=mu, scale=alpha * np.sqrt(2))
    # return cauchy.pdf(x, loc=mu, scale=alpha)
    # return skewnorm.pdf(x, skew, loc=mu, scale=alpha)
    # return exponnorm.pdf(x, beta, loc=mu, scale=alpha)
    # l2 = np.abs(l2)
    # x = (x - mu) / alpha
    # return 2 / alpha * norm.pdf(x) * norm.cdf(l1 * x / np.sqrt(1 + l2 * x ** 2))


if __name__ == "__main__":
    # Define the location of all your files
    # this will put everything into the example dir

    examples_dir = dirname(realpath(__file__))
    data_dir = join(examples_dir, "data")
    image_dir = join(examples_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    in_file = os.path.join(
        examples_dir,
        f"results/HD_22049_mask_new_out_monh_teff_logg_vmic_vmac_vsini.sme",
    )
    sme = SME.SME_Structure.load(in_file)

    segments = np.arange(6, 31)
    mask = sme.mask_good[segments]
    resid = sme.fitresults.residuals
    unc = np.concatenate(sme.uncs[segments][mask])

    for i, param in enumerate(sme.fitresults.parameters):
        pder = sme.fitresults.derivative[:, i]
        gradlim = np.median(np.abs(pder))
        idx = np.abs(pder) > gradlim

        # Sort pixels according to the change of the i
        # parameter needed to match the observations
        idx_sort = np.argsort(resid[idx] / pder[idx])
        ch_x = resid[idx][idx_sort] / pder[idx][idx_sort]
        # Weights of the individual pixels also sorted
        ch_y = np.abs(pder[idx][idx_sort]) / unc[idx][idx_sort]
        # Cumulative weights
        ch_y = np.cumsum(ch_y)
        # Normalized cumulative weights
        ch_y /= ch_y[-1]

        # Initial guess

        hmed = np.interp(0.5, ch_y, ch_x)
        interval = np.interp([0.16, 0.84], ch_y, ch_x)
        sigma_estimate = (interval[1] - interval[0]) / 2

        p0 = [hmed, sigma_estimate]
        sopt = p0

        # try:
        #     sopt, _ = curve_fit(pdf, b, h, p0=p0)
        # except RuntimeError:
        #     try:
        #         sopt, _ = curve_fit(pdf, b, h, p0=p0, method="dogbox")
        #     except RuntimeError:
        #         sopt = p0

        # Fit the distribution
        # try:
        #     sopt, _ = curve_fit(cdf, ch_x, ch_y, p0=p0)
        # except RuntimeError:
        #     # Fit failed, use dogbox instead
        #     try:
        #         sopt, _ = curve_fit(cdf, ch_x, ch_y, p0=p0, method="dogbox")
        #     except RuntimeError:
        #         sopt = p0

        # sigma, interval = std(*sopt)
        # hmed = (interval[0] + interval[1]) / 2

        # Plot 1 (cumulative distribution)
        r = (sopt[0] - 3 * sopt[1], sopt[0] + 3 * sopt[1])
        x = np.linspace(ch_x.min(), ch_x.max(), ch_x.size * 10)
        plt.plot(ch_x, ch_y, "+", label="measured")
        plt.plot(x, cdf(x, *sopt), label="Gaussian")

        plt.hlines(0.5, r[0], r[1], linestyles="dashed")
        plt.vlines(hmed, -0.1, 1.1, linestyles="dashed")
        plt.hlines([0.16, 0.84], r[0], r[1], linestyles=["dashdot"])
        plt.vlines(interval, -0.1, 1.1, linestyles=["dashdot"])

        plt.xlabel(f"$\Delta${param}")
        plt.ylabel("cumulative probability")
        plt.title(f"Cumulative Probability: {param}")
        plt.xlim(r[0], r[1])
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.savefig(join(image_dir, f"cumulative_probability_{param}.png"))
        plt.clf()
        # plt.show()
        # Plot 2 (density distribution)
        # r = (sopt[0] - 20 * sopt[1], sopt[0] + 20 * sopt[1])
        x = np.linspace(r[0], r[-1], ch_x.size * 10)
        h, b, _ = plt.hist(
            ch_x, bins="auto", density=True, histtype="step", range=r, label="measured",
        )
        where = (b[:-1] > interval[0]) & (b[:-1] < interval[1])
        plt.fill_between(
            b[:-1], h, where=where, interpolate=False, step="post", alpha=0.5
        )
        plt.plot(x, pdf(x, *sopt), label="Gaussian")
        # plt.plot(x, pdf(x, *p0), "--", label="estimate")

        idx = np.argmin(np.abs(b - hmed))
        plt.vlines(hmed, 0, h[idx], linestyles="dashed")
        plt.vlines(b[:-1][where][0], 0, h[where][0], linestyles="dashdot")
        plt.vlines(b[:-1][where][-1], 0, h[where][-1], linestyles="dashdot")

        plt.xlabel(f"$\Delta${param}")
        plt.ylabel("probability")
        plt.xlim(r)
        plt.title(f"Probability Density: {param}")
        plt.legend()
        plt.savefig(join(image_dir, f"probability_density_{param}.png"))
        plt.clf()
        # plt.show()
        pass

    print(f"Finished")
