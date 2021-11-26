# -*- coding: utf-8 -*-
""" Minimum working example of an SME script
"""
import datetime
import os
import os.path
import re
from os.path import dirname, join, realpath

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import coordinates as coord
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from data_sources.StellarDB import StellarDB
from scipy.constants import speed_of_light
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.linalg import lstsq, solve_banded
from scipy.ndimage.filters import gaussian_filter1d, median_filter
from scipy.optimize import least_squares, minimize
from scipy.optimize.minpack import curve_fit
from scipy.stats import exponnorm, gennorm, norm, skewnorm
from tqdm import tqdm

from pysme import sme as SME
from pysme import util
from pysme.abund import Abund
from pysme.continuum_and_radial_velocity import determine_radial_velocity
from pysme.gui import plot_plotly
from pysme.iliffe_vector import Iliffe_vector
from pysme.linelist.vald import ValdFile
from pysme.persistence import save_as_idl
from pysme.solve import solve
from pysme.synthesize import synthesize_spectrum

clight = speed_of_light * 1e-3


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
    """1 sigma (68.27 %) quantile, assuming symmetric distribution"""
    # percentile = 0.997  # 5 sigma
    # interval = gennorm.interval(percentile, beta, loc=mu, scale=alpha * np.sqrt(2))
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


def fit(x):
    plimit = np.clip(x[0], 0, 100)
    gradlim = np.nanpercentile(np.abs(pder), plimit)
    idx = pder != 0
    idx &= np.abs(resid) < 5 * unc
    idx &= np.abs(pder) < gradlim

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

    p0 = [hmed, sigma_estimate, 1]
    sopt = p0

    # Fit the distribution
    try:
        sopt, _ = curve_fit(cdf, ch_x, ch_y, p0=p0)
    except RuntimeError:
        # Fit failed, use dogbox instead
        try:
            sopt, _ = curve_fit(cdf, ch_x, ch_y, p0=p0, method="dogbox")
        except RuntimeError:
            sopt = p0

    sigma, interval = std(*sopt)
    hmed = (interval[0] + interval[1]) / 2

    res = ch_y - cdf(ch_x, *sopt)
    return np.sum(np.nan_to_num(res) ** 2) / ch_y.size


if __name__ == "__main__":
    # Define the location of all your files
    # this will put everything into the example dir

    examples_dir = dirname(realpath(__file__))
    data_dir = join(examples_dir, "data")
    image_dir = join(examples_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    in_file = os.path.join(
        examples_dir, f"results/Eps_Eri_monh_teff_logg_vmic_vmac_vsini.sme"
    )
    sme = SME.SME_Structure.load(in_file)

    segments = np.arange(6, 31)
    mask = sme.mask_good[segments]
    resid = sme.fitresults.residuals
    unc = np.concatenate(sme.uncs[segments][mask])
    spec = np.concatenate(sme.spec[segments][mask])
    # unc = np.sqrt(s)
    # unc = np.full(resid.size, 1)

    values = sme.fitresults.values
    residuals = sme.fitresults.residuals
    derivative = sme.fitresults.derivative

    # For monh
    ratio = np.abs(derivative / values)
    median = np.median(ratio, axis=0)
    idx_param = (ratio > median)[:, 0]
    chi2_param = np.nanmean((residuals[idx_param] / unc[idx_param]) ** 2)

    limit = np.median(np.abs(derivative / values))
    chi2 = np.sum(residuals ** 2) / (residuals.size - len(values))
    # unc *= np.sqrt(chi2)

    jac = derivative / np.sqrt(chi2)
    _, s, VT = np.linalg.svd(jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[: s.size]
    pcov = np.dot(VT.T / s ** 2, VT)
    uncertainties = np.sqrt(np.diag(pcov))

    # def gauss(x, A, sig, mu, B):
    #     return B - A * np.exp(-((x - mu) ** 2) / (2 * sig ** 2))

    # width = 100
    # wave = sme.wave.ravel()
    # spec = sme.spec.ravel()
    # wave_lines = sme.linelist.wlcent[sme.linelist.depth > 0.1]
    # wave_lines *= 1 + sme.vrad[10] / clight
    # idx_wave = np.digitize(wave_lines, wave)
    # idx_lines = idx_wave[:, None] + np.arange(-width, width)[None, :]
    # lines = spec[idx_lines]

    # x = np.arange(width * 2)
    # B0 = np.max(lines, axis=1)
    # A0 = B0 - np.min(lines, axis=1)
    # sig0 = np.full(lines.shape[0], 10)
    # mu0 = np.full(lines.shape[0], width)

    # n = lines.shape[0]
    # for i in tqdm(range(n)):
    #     res = least_squares(
    #         lambda p: (gauss(x, p[0], p[1], p[2], p[3]) - lines[i]),
    #         x0=[A0[i], sig0[i], mu0[i], B0[i]],
    #         method="lm",
    #     )
    #     A0[i] = res.x[0]
    #     sig0[i] = res.x[1]
    #     mu0[i] = res.x[2]
    #     B0[i] = res.x[3]

    # fn = gauss(x[None, :], 1, sig0[:, None], width, 1)
    # line = np.nanmedian(fn, axis=0)
    # line_mad = np.nanmedian(np.abs(fn - line), axis=0)
    # line_std = line_mad * 1.48

    # plt.fill_between(x, line - line_std, line + line_std, alpha=0.5)
    # plt.plot(line)
    # plt.show()

    for i, param in enumerate(sme.fitresults.parameters):
        pder = sme.fitresults.derivative[:, i] / np.sqrt(chi2)
        pval = sme.fitresults.values[i]

        idx = pder != 0
        # The original rule used in the paper
        # idx &= np.abs(pder) < np.percentile(np.abs(pder), 84)
        # idx &= np.abs(pder) > np.percentile(np.abs(pder), 16)
        # idx &= np.abs(residuals) < 5 * np.std(residuals)

        # idx &= np.abs(resid) < 5 * unc
        # idx &= np.abs(pder) < limit * np.abs(pval)
        # idx &= np.abs(pder) < 5 * np.median(np.abs(resid))
        # Fit which cutoff makes the curve most gaussian
        # res = minimize(fit, [80], method="Nelder-Mead")
        # plimit = res.x[0]
        # gradlim = np.nanpercentile(np.abs(pder), plimit)
        # idx &= np.abs(pder) < gradlim
        # Only use the center part of ch_x
        # ch_x = resid / pder
        # plimit = np.nanpercentile(ch_x, [45, 55])
        # idx &= (ch_x > plimit[0]) & (ch_x < plimit[1])

        # Only use derivatives around the center
        # this is roughly equivalent to the above cutoff
        # due to the percentile
        # med = np.median(np.abs(pder))
        # mad = np.median(np.abs(np.abs(pder) - med))
        # idx &= np.abs(pder) < med + 20 * mad

        percentage_points = np.count_nonzero(idx) / idx.size * 100
        print(f"Using {percentage_points:.2f}% points for the derivative")

        # Sort pixels according to the change of the i
        # parameter needed to match the observations
        idx_sort = np.argsort(resid[idx] / pder[idx])
        ch_x = resid[idx][idx_sort] / pder[idx][idx_sort]
        # Weights of the individual pixels also sorted
        ch_y = np.abs(pder[idx][idx_sort])  # / unc[idx][idx_sort]
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

        # # Fit the distribution
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
        sigma = sigma_estimate

        # Plot 1 (cumulative distribution)
        r = (hmed - 3 * sigma, hmed + 3 * sigma)
        x = np.linspace(r[0], r[1], ch_x.size * 10)
        plt.plot(ch_x, ch_y, "+", label="measured")
        plt.plot(x, cdf(x, *sopt), label="Gaussian")

        plt.hlines(0.5, r[0], r[1], linestyles="dashed")
        plt.vlines(hmed, -0.1, 1.1, linestyles="dashed")
        plt.hlines([0.16, 0.84], r[0], r[1], linestyles=["dashdot"])
        plt.vlines(interval, -0.1, 1.1, linestyles=["dashdot"])

        plt.xlabel(fr"$\Delta${param}")
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
            ch_x,
            bins="auto",
            density=True,
            histtype="step",
            range=r,
            label="measured",
        )
        where = (b[:-1] > interval[0]) & (b[:-1] < interval[1])
        plt.fill_between(
            b[:-1], h, where=where, interpolate=False, step="post", alpha=0.5
        )
        plt.plot(x, pdf(x, hmed, sigma), label="Gaussian")
        # plt.plot(x, pdf(x, *p0), "--", label="estimate")

        idx = np.argmin(np.abs(b - hmed))
        plt.vlines(hmed, 0, h[idx], linestyles="dashed")
        plt.vlines(b[:-1][where][0], 0, h[where][0], linestyles="dashdot")
        plt.vlines(b[:-1][where][-1], 0, h[where][-1], linestyles="dashdot")

        plt.xlabel(fr"$\Delta${param}")
        plt.ylabel("probability")
        plt.xlim(r)
        plt.title(f"Probability Density: {param}")
        plt.legend()
        plt.savefig(join(image_dir, f"probability_density_{param}.png"))
        plt.clf()
        # plt.show()
        pass

    print(f"Finished")
