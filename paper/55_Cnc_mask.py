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
from astropy.io import fits
from astropy.time import Time
from data_sources.StellarDB import StellarDB
from scipy.linalg import lstsq, solve_banded
from scipy.ndimage.filters import gaussian_filter1d, median_filter
from scipy.optimize import least_squares
from tqdm import tqdm

from pysme import sme as SME
from pysme import util
from pysme.abund import Abund
from pysme.gui import plot_plotly
from pysme.iliffe_vector import Iliffe_vector
from pysme.linelist.vald import ValdFile
from pysme.persistence import save_as_idl
from pysme.solve import solve
from pysme.synthesize import synthesize_spectrum


def get_teff_from_spectral_type(spectral_type):
    spectral_class = spectral_type[0]
    spectral_number = spectral_type[1]

    teff_dict = {
        "M": {
            "0": 3800,
            "1": 3600,
            "2": 3400,
            "3": 3250,
            "4": 3100,
            "5": 2800,
            "6": 2600,
            "7": 2500,
            "8": 2400,
            "9": 2300,
        }
    }
    return teff_dict[spectral_class][spectral_number]


def black_body_curve(teff, wave):
    h = const.h.to_value("J s")
    c = const.c.to_value("AA/s")
    kB = const.k_B.to_value("J/K")
    B = 2 * h * c ** 2 / wave ** 5 * (np.exp(h * c / (wave * kB * teff)) - 1) ** -1
    return B


def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s)
        # pre-sorting of locals min based on relative position with respect to s_mid
        lmin = lmin[s[lmin] < s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid
        lmax = lmax[s[lmax] > s_mid]

    # global max of dmax-chunks of locals max
    lmin = lmin[
        [i + np.argmin(s[lmin[i : i + dmin]]) for i in range(0, len(lmin), dmin)]
    ]
    # global min of dmin-chunks of locals min
    lmax = lmax[
        [i + np.argmax(s[lmax[i : i + dmax]]) for i in range(0, len(lmax), dmax)]
    ]

    return lmin, lmax


class Plot_Normalization:  # pragma: no cover
    def __init__(self, wsort, sB, new_wave, contB, iteration=0, title=None):
        plt.ion()
        self.fig = plt.figure()
        self.title = title
        suptitle = f"Iteration: {iteration}"
        if self.title is not None:
            suptitle = f"{self.title}\n{suptitle}"
        self.fig.suptitle(suptitle)

        self.ax = self.fig.add_subplot(111)
        self.line1 = self.ax.plot(wsort, sB, label="Spectrum")[0]
        self.line2 = self.ax.plot(new_wave, contB, label="Continuum Fit")[0]
        plt.legend()

        plt.show()

    def plot(self, wsort, sB, new_wave, contB, iteration):
        suptitle = f"Iteration: {iteration}"
        if self.title is not None:
            suptitle = f"{self.title}\n{suptitle}"
        self.fig.suptitle(suptitle)

        self.line1.set_xdata(wsort)
        self.line1.set_ydata(sB)
        self.line2.set_xdata(new_wave)
        self.line2.set_ydata(contB)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        plt.ioff()
        plt.close()


def opt_filter(y, par, par1=None, weight=None, lambda2=-1, maxiter=100):
    """
    Optimal filtering of 1D and 2D arrays.
    Uses tridiag in 1D case and sprsin and linbcg in 2D case.
    Written by N.Piskunov 8-May-2000

    Parameters
    ----------
    f : array
        1d or 2d array
    xwidth : int
        filter width (for 2d array width in x direction (1st index)
    ywidth : int
        (for 2d array only) filter width in y direction (2nd index) if ywidth is missing for 2d array, it set equal to xwidth
    weight : array(float)
        an array of the same size(s) as f containing values between 0 and 1
    lambda1: float
        regularization parameter
    maxiter : int
        maximum number of iteration for filtering of 2d array
    """

    y = np.asarray(y)

    if y.ndim not in [1, 2]:
        raise ValueError("Input y must have 1 or 2 dimensions")

    if par < 1:
        par = 1

    # 1D case
    if y.ndim == 1 or (y.ndim == 2 and (y.shape[0] == 1 or y.shape[1] == 1)):
        y = y.ravel()
        n = y.size

        if weight is None:
            weight = np.ones(n)
        elif np.isscalar(weight):
            weight = np.full(n, weight)
        else:
            weight = weight[:n]

        if lambda2 > 0:
            # Apply regularization lambda
            aij = np.zeros((5, n))
            # 2nd lower subdiagonal
            aij[0, 2:] = lambda2
            # Lower subdiagonal
            aij[1, 1] = -par - 2 * lambda2
            aij[1, 2:-1] = -par - 4 * lambda2
            aij[1, -1] = -par - 2 * lambda2
            # Main diagonal
            aij[2, 0] = weight[0] + par + lambda2
            aij[2, 1] = weight[1] + 2 * par + 5 * lambda2
            aij[2, 2:-2] = weight[2:-2] + 2 * par + 6 * lambda2
            aij[2, -2] = weight[-2] + 2 * par + 5 * lambda2
            aij[2, -1] = weight[-1] + par + lambda2
            # Upper subdiagonal
            aij[3, 0] = -par - 2 * lambda2
            aij[3, 1:-2] = -par - 4 * lambda2
            aij[3, -2] = -par - 2 * lambda2
            # 2nd lower subdiagonal
            aij[4, 0:-2] = lambda2
            # RHS
            b = weight * y

            f = solve_banded((2, 2), aij, b)
        else:
            a = np.full(n, -abs(par))
            b = np.copy(weight) + abs(par)
            b[1:-1] += abs(par)
            aba = np.array([a, b, a])

            f = solve_banded((1, 1), aba, weight * y)

        return f
    else:
        # 2D case
        if par1 is None:
            par1 = par
        if par == 0 and par1 == 0:
            raise ValueError("xwidth and ywidth can't both be 0")
        n = y.size
        nx, ny = y.shape

        lam_x = abs(par)
        lam_y = abs(par1)

        n = nx * ny
        ndiag = 2 * nx + 1
        aij = np.zeros((n, ndiag))
        aij[nx, 0] = weight[0, 0] + lam_x + lam_y
        aij[nx, 1:nx] = weight[0, 1:nx] + 2 * lam_x + lam_y
        aij[nx, nx : n - nx] = weight[1 : ny - 1] + 2 * (lam_x + lam_y)
        aij[nx, n - nx : n - 1] = weight[ny - 1, 0 : nx - 1] + 2 * lam_x + lam_y
        aij[nx, n - 1] = weight[ny - 1, nx - 1] + lam_x + lam_y

        aij[nx - 1, 1:n] = -lam_x
        aij[nx + 1, 0 : n - 1] = -lam_x

        ind = np.arrange(ny - 1) * nx + nx + nx * n
        aij[ind - 1] = aij[ind - 1] - lam_x
        aij[ind] = aij[ind] - lam_x

        ind = np.arrange(ny - 1) * nx + nx
        aij[nx + 1, ind - 1] = 0
        aij[nx - 1, ind] = 0

        aij[0, nx:n] = -lam_y
        aij[ndiag - 1, 0 : n - nx] = -lam_y

        rhs = f * weight

        model = solve_banded((nx, nx), aij, rhs)
        model = np.reshape(model, (ny, nx))
        return model


def middle(
    f,
    param,
    x=None,
    iterations=40,
    eps=0.001,
    poly=False,
    weight=1,
    lambda2=-1,
    mn=None,
    mx=None,
):
    """
    middle tries to fit a smooth curve that is located
    along the "middle" of 1D data array f. Filter size "filter"
    together with the total number of iterations determine
    the smoothness and the quality of the fit. The total
    number of iterations can be controlled by limiting the
    maximum number of iterations (iter) and/or by setting
    the convergence criterion for the fit (eps)
    04-Nov-2000 N.Piskunov wrote.
    09-Nov-2011 NP added weights and 2nd derivative constraint as LAM2

    Parameters
    ----------
    f : Callable
        Function to fit
    filter : int
        Smoothing parameter of the optimal filter (or polynomial degree of poly is True)
    iter : int
        maximum number of iterations [def: 40]
    eps : float
        convergence level [def: 0.001]
    mn : float
        minimum function values to be considered [def: min(f)]
    mx : float
        maximum function values to be considered [def: max(f)]
    lam2 : float
        constraint on 2nd derivative
    weight : array(float)
        vector of weights.
    """
    mn = mn if mn is not None else np.min(f)
    mx = mx if mx is not None else np.max(f)

    f = np.asarray(f)

    if x is None:
        xx = np.linspace(-1, 1, num=f.size)
    else:
        xx = np.asarray(x)

    if poly:
        j = (f >= mn) & (f <= mx)
        n = np.count_nonzero(j)
        if n <= round(param):
            return f

        fmin = np.min(f[j]) - 1
        fmax = np.max(f[j]) + 1
        ff = (f[j] - fmin) / (fmax - fmin)
        ff_old = ff
    else:
        fmin = np.min(f) - 1
        fmax = np.max(f) + 1
        ff = (f - fmin) / (fmax - fmin)
        ff_old = ff
        n = len(f)

    for _ in range(iterations):
        if poly:
            param = round(param)
            if param > 0:
                t = median_filter(np.polyval(np.polyfit(xx, ff, param), xx), 3)
                tmp = np.polyval(np.polyfit(xx, (t - ff) ** 2, param), xx)
            else:
                t = np.tile(np.polyfit(xx, ff, param), len(f))
                tmp = np.tile(np.polyfit(xx, (t - ff) ** 2, param), len(f))
        else:
            t = median_filter(opt_filter(ff, param, weight=weight, lambda2=lambda2), 3)
            tmp = opt_filter(
                weight * (t - ff) ** 2, param, weight=weight, lambda2=lambda2
            )

        dev = np.sqrt(np.clip(tmp, 0, None))
        ff = np.clip(t - dev, ff, t + dev)
        dev2 = np.max(weight * np.abs(ff - ff_old))
        ff_old = ff

        # print(dev2)
        if dev2 <= eps:
            break

    if poly:
        xx = np.linspace(-1, 1, len(f))
        if param > 0:
            t = median_filter(np.polyval(np.polyfit(xx, ff, param), xx), 3)
        else:
            t = np.tile(np.polyfit(xx, ff, param), len(f))

    return t * (fmax - fmin) + fmin


def top(
    f,
    order=1,
    iterations=40,
    eps=0.001,
    poly=False,
    weight=1,
    lambda2=-1,
    mn=None,
    mx=None,
):
    """
    top tries to fit a smooth curve to the upper envelope
    of 1D data array f. Filter size "filter"
    together with the total number of iterations determine
    the smoothness and the quality of the fit. The total
    number of iterations can be controlled by limiting the
    maximum number of iterations (iter) and/or by setting
    the convergence criterion for the fit (eps)
    04-Nov-2000 N.Piskunov wrote.
    09-Nov-2011 NP added weights and 2nd derivative constraint as LAM2

    Parameters
    ----------
    f : Callable
        Function to fit
    filter : int
        Smoothing parameter of the optimal filter (or polynomial degree of poly is True)
    iter : int
        maximum number of iterations [def: 40]
    eps : float
        convergence level [def: 0.001]
    mn : float
        minimum function values to be considered [def: min(f)]
    mx : float
        maximum function values to be considered [def: max(f)]
    lam2 : float
        constraint on 2nd derivative
    weight : array(float)
        vector of weights.
    """
    mn = mn if mn is not None else np.min(f)
    mx = mx if mx is not None else np.max(f)

    f = np.asarray(f)
    xx = np.linspace(-1, 1, num=f.size)

    if poly:
        j = (f >= mn) & (f <= mx)
        if np.count_nonzero(j) <= round(order):
            raise ValueError("Not enough points")
        fmin = np.min(f[j]) - 1
        fmax = np.max(f[j]) + 1
        ff = (f - fmin) / (fmax - fmin)
        ff_old = ff
    else:
        fff = middle(
            f,
            order,
            iterations=iterations,
            eps=eps,
            weight=weight,
            lambda2=lambda2,
        )
        fmin = np.min(f) - 1
        fmax = np.max(f) + 1
        fff = (fff - fmin) / (fmax - fmin)
        ff = (f - fmin) / (fmax - fmin) / fff
        ff_old = ff

    for _ in range(iterations):
        order = round(order)
        if poly:
            t = median_filter(np.polyval(np.polyfit(xx, ff, order), xx), 3)
            tmp = np.polyval(np.polyfit(xx, np.clip(ff - t, 0, None) ** 2, order), xx)
            dev = np.sqrt(np.clip(tmp, 0, None))
        else:
            t = median_filter(opt_filter(ff, order, weight=weight, lambda2=lambda2), 3)
            tmp = opt_filter(
                np.clip(weight * (ff - t), 0, None),
                order,
                weight=weight,
                lambda2=lambda2,
            )
            dev = np.sqrt(np.clip(tmp, 0, None))

        ff = np.clip(t - eps, ff, t + dev * 3)
        dev2 = np.max(weight * np.abs(ff - ff_old))
        ff_old = ff
        if dev2 <= eps:
            break

    if poly:
        t = median_filter(np.polyval(np.polyfit(xx, ff, order), xx), 3)
        return t * (fmax - fmin) + fmin
    else:
        return t * fff * (fmax - fmin) + fmin


def continuum_normalize(
    spec,
    wave,
    cont,
    sigm,
    iterations=10,
    smooth_initial=1e5,
    smooth_final=5e6,
    scale_vert=1,
    plot=True,
    plot_title=None,
):
    """Fit a continuum to a spectrum by slowly approaching it from the top.
    We exploit here that the continuum varies only on large wavelength scales, while individual lines act on much smaller scales

    TODO automatically find good parameters for smooth_initial and smooth_final
    TODO give variables better names

    Parameters
    ----------
    spec : masked array of shape (nord, ncol)
        Observed input spectrum, masked values describe column ranges
    wave : masked array of shape (nord, ncol)
        Wavelength solution of the spectrum
    cont : masked array of shape (nord, ncol)
        Initial continuum guess, for example based on the blaze
    sigm : masked array of shape (nord, ncol)
        Uncertainties of the spectrum
    iterations : int, optional
        Number of iterations of the algorithm,
        note that runtime roughly scales with the number of iterations squared
        (default: 10)
    smooth_initial : float, optional
        Smoothing parameter in the initial runs, usually smaller than smooth_final (default: 1e5)
    smooth_final : float, optional
        Smoothing parameter of the final run (default: 5e6)
    scale_vert : float, optional
        Vertical scale of the spectrum. Usually 1 if a previous normalization exists (default: 1)
    plot : bool, optional
        Wether to plot the current status and results or not (default: True)

    Returns
    -------
    cont : masked array of shape (nord, ncol)
        New continuum
    """

    nord, ncol = spec.shape

    par2 = 1e-4
    par4 = 0.01 * (1 - np.clip(2, None, 1 / np.sqrt(np.ma.median(spec))))

    b = np.clip(cont, 1, None)
    mask = ~np.ma.getmaskarray(b)
    for i in range(nord):
        b[i, mask[i]] = middle(b[i, mask[i]], 1)
    cont = b

    # Create new equispaced wavelength grid
    tmp = np.ma.compressed(wave)
    wmin = np.min(tmp)
    wmax = np.max(tmp)
    dwave = np.abs(tmp[tmp.size // 2] - tmp[tmp.size // 2 - 1]) * 0.5
    nwave = np.ceil((wmax - wmin) / dwave) + 1
    new_wave = np.linspace(wmin, wmax, int(nwave), endpoint=True)

    # Combine all orders into one big spectrum, sorted by wavelength
    wsort, j, index = np.unique(tmp, return_index=True, return_inverse=True)
    sB = np.ma.compressed(spec / cont)[j]

    # Get initial weights for each point
    weight = middle(sB, 0.5, x=wsort - wmin)
    weight = weight / middle(weight, 3 * smooth_initial) + np.concatenate(
        ([0], 2 * weight[1:-1] - weight[0:-2] - weight[2:], [0])
    )
    weight = np.clip(weight, 0, None)
    # TODO for some reason the interpolation messes up, use linear instead for now
    # weight = util.safe_interpolation(wsort, weight, new_wave)
    weight = np.interp(new_wave, wsort, weight)
    weight /= np.max(weight)

    # Interpolate Spectrum onto the new grid
    # ssB = util.safe_interpolation(wsort, sB, new_wave)
    ssB = np.interp(new_wave, wsort, sB)
    # Keep the scale of the continuum
    bbb = middle(np.ma.compressed(cont)[j], 1)

    contB = np.ones_like(ssB)
    if plot:  # pragma: no cover
        p = Plot_Normalization(wsort, sB, new_wave, contB, 0, title=plot_title)

    try:
        for i in tqdm(range(iterations)):
            # Find new approximation of the top, smoothed by some parameter
            c = ssB / contB
            for _ in range(iterations):
                _c = top(
                    c,
                    smooth_initial,
                    eps=par2,
                    weight=weight,
                    lambda2=smooth_final,
                )
                c = np.clip(_c, c, None)
            c = (
                top(
                    c,
                    smooth_initial,
                    eps=par4,
                    weight=weight,
                    lambda2=smooth_final,
                )
                * contB
            )

            # Scale it and update the weights of each point
            contB = c * scale_vert
            contB = middle(contB, 1)
            weight = np.clip(ssB / contB, None, contB / np.clip(ssB, 1, None))

            # Plot the intermediate results
            if plot:  # pragma: no cover
                p.plot(wsort, sB, new_wave, contB, i)
    except ValueError:
        # logger.error("Continuum fitting aborted")
        pass
    finally:
        if plot:  # pragma: no cover
            p.close()

    # Calculate the new continuum from intermediate values
    # new_cont = util.safe_interpolation(new_wave, contB, wsort)
    new_cont = np.interp(wsort, new_wave, contB)
    mask = np.ma.getmaskarray(cont)
    cont[~mask] = (new_cont * bbb)[index]

    # Final output plot
    if plot:  # pragma: no cover
        plt.plot(wave.ravel(), spec.ravel(), label="spec")
        plt.plot(wave.ravel(), cont.ravel(), label="cont")
        plt.legend(loc="best")
        if plot_title is not None:
            plt.title(plot_title)
        plt.xlabel("Wavelength [A]")
        plt.ylabel("Flux")
        plt.show()

    return cont


def get_mask_from_neural_network(sme):
    from continuum_neural_network import ContinuumModel

    model = ContinuumModel()
    model.load("model.dat")
    npoints_model = model.model.input_shape[-1]

    mask = [None for _ in sme.spec]

    for i, spec in enumerate(sme.spec):
        npoints_seg = len(spec)
        nswaths = int(np.ceil(npoints_seg) / npoints_model)
        idx_seg = np.arange(npoints_seg)
        mask[i] = np.full(npoints_seg, sme.mask_values["line"])
        for j in range(nswaths - 1):
            low, high = j * npoints_model, (j + 1) * npoints_model
            idx = idx_seg[low:high]
            X = spec[low:high][None, :]
            y = model.predict(X)[0]
            mask[i][low:high] = np.where(
                y > 0.5, sme.mask_values["continuum"], sme.mask_values["line"]
            )
        low, high = npoints_seg - npoints_model, npoints_seg
        idx = idx_seg[low:high]
        X = spec[low:high][None, :]
        y = model.predict(X)[0]
        mask[i][low:high] = np.where(
            y > 0.5, sme.mask_values["continuum"], sme.mask_values["line"]
        )

        m = mask[i] == sme.mask_values["continuum"]
        # plt.plot(idx_seg, spec)
        # plt.plot(idx_seg[m], spec[m], "+")
        # plt.show()
        pass

    return mask


if __name__ == "__main__":
    # Define the location of all your files
    # this will put everything into the example dir
    target = "55_Cnc"
    star = StellarDB().load(target)
    alias = [re.sub(r"[-_ ]", "", s).lower() for s in star["id"]]

    examples_dir = dirname(realpath(__file__))
    data_dir = join(examples_dir, "data")

    # Find the correct data file for this target
    # fname = "ADP.2019-01-30T01:13:58.172.fits"
    fname = "55_Cnc_mask.sme"
    in_file = os.path.join(examples_dir, "results", fname)
    # in_file = os.path.join(examples_dir, f"results/{target}_mask.sme")

    vald_file = os.path.join(examples_dir, f"data/hd22049.lin")

    out_file = os.path.join(examples_dir, f"results/{target}_mask_out.sme")
    plot_file = os.path.join(examples_dir, f"results/{target}.html")
    date_string = datetime.datetime.now().isoformat().replace(":", ".")
    log_file = os.path.join(examples_dir, f"results/{target}_{date_string}.log")

    # Start the logging to the file
    util.start_logging(log_file)

    # err = hdu[1].data["ERR"]
    sme = SME.SME_Structure.load(in_file)
    sme.mu = np.geomspace(0.1, 1, num=7)

    # sme.uncs = [
    #     np.nan_to_num(1 / np.sqrt(np.abs(spec)) ** 2, nan=1) for spec in sme.spec
    # ]
    # # sme.mask = get_mask_from_neural_network(sme)
    # # sme.mask = sme.mask_values["line"]
    # # for i in range(sme.nseg):
    # #     sme.mask[i][sme.mask[i] == 0] = sme.mask_values["bad"]

    # # Add telluric data (without rayleigh scattering)
    # ftapas = [np.interp(w, wtapas, ftapas) for w in wave]
    # sme.telluric = Iliffe_vector(values=ftapas)

    # Get first guess from literature values
    sme.teff = 5065  # star["t_eff"].to_value("K") if "t_eff" in star else 6000
    sme.logg = 4.42  # star["logg"].to_value(1) if "logg" in star else 4
    monh = 0.35  # star["metallicity"].to_value(1) if "metallicity" in star else 0
    sme.abund = Abund(monh, "asplund2009")
    # sme.vmic = (
    #     star["velocity_turbulence"].to_value("km/s")
    #     if "velocity_turbulence" in star
    #     else 3
    # )
    # Test this
    sme.vmic = 1
    sme.vmac = 2
    sme.vsini = 1.23

    # load the linelist
    sme.linelist = ValdFile(vald_file)

    # Set the atmosphere grid
    sme.atmo.source = "marcs2014.sav"
    sme.atmo.geom = "PP"

    # Add NLTE corrections
    sme.nlte.set_nlte("Al", "nlte_Al_ama51_pysme.grd")
    sme.nlte.set_nlte("Ba", "nlte_Ba_ama51_pysme.grd")
    sme.nlte.set_nlte("Ca", "nlte_Ca_ama51_pysme.grd")
    sme.nlte.set_nlte("C", "nlte_C_ama51_pysme.grd")
    sme.nlte.set_nlte("H", "nlte_H_ama51_pysme.grd")
    sme.nlte.set_nlte("K", "nlte_K_ama51_pysme.grd")
    sme.nlte.set_nlte("Li", "nlte_Li_ama51_pysme.grd")
    sme.nlte.set_nlte("Mg", "nlte_Mg_ama51_pysme.grd")
    sme.nlte.set_nlte("Mn", "nlte_Mn_ama51_pysme.grd")
    sme.nlte.set_nlte("Na", "nlte_Na_ama51_pysme.grd")
    sme.nlte.set_nlte("N", "nlte_Na_ama51_pysme.grd")
    sme.nlte.set_nlte("O", "nlte_O_ama51_pysme.grd")
    sme.nlte.set_nlte("Si", "nlte_Si_ama51_pysme.grd")
    sme.nlte.set_nlte("Fe", "marcs2012_Fe2016.grd")

    # # Barycentric correction
    # obstime = Time(hdu[0].header["DATE-OBS"])
    # obs_long = hdu[0].header["HIERARCH ESO TEL GEOLON"]
    # obs_lat = hdu[0].header["HIERARCH ESO TEL GEOLAT"]
    # obs_alt = hdu[0].header["HIERARCH ESO TEL GEOELEV"]
    # observatory = coord.EarthLocation.from_geodetic(obs_long, obs_lat, height=obs_alt)
    # sky_location = star["coordinates"]
    # sky_location.obstime = obstime
    # sky_location.location = observatory
    # correction = sky_location.radial_velocity_correction().to_value("km/s")

    # Set radial velocity and continuum settings
    # Set RV and Continuum flags
    sme.vrad_flag = "each"
    sme.cscale_flag = 2
    sme.cscale_type = "match+mask"

    # sme.vrad = (
    #     star["radial_velocity"].to_value("km/s") if "radial_velocity" in star else 0
    # )
    # sme.vrad -= correction
    # checked manually
    # sme.vrad = 26.3

    # Define any fitparameters you want
    # For abundances use: 'abund {El}', where El is the element (e.g. 'abund Fe')
    # For linelist use: 'linelist {Nr} {p}', where Nr is the number in the
    # linelist and p is the line parameter (e.g. 'linelist 17 gflog')
    fitparameters = [
        ["monh"],
        ["monh", "teff", "logg", "vmic", "vmac", "vsini"],
    ]
    # Restrict the linelist to relevant lines
    # for this segment
    rvel = 100
    wmin, wmax = sme.wran[6][0], sme.wran[30][1]
    wmin *= 1 - rvel / 3e5
    wmax *= 1 + rvel / 3e5
    sme.linelist = sme.linelist.trim(wmin, wmax)

    # Start SME solver
    sme.cscale = [0, 0, 1]
    # sme = synthesize_spectrum(sme, segments=np.arange(6, 31))

    mask_file = os.path.join(
        examples_dir,
        f"results/HD_22049_mask_2_out_monh_teff_logg_vmic_vmac_vsini.sme",
    )
    sme_mask = SME.SME_Structure.load(mask_file)
    sme = sme.import_mask(sme_mask)
    # wave = sme_mask.wave.ravel()
    # telluric = sme_mask.telluric.ravel()
    # sme.telluric = [np.interp(sme.wave[i], wave, telluric) for i in range(sme.nseg)]
    # for i in range(sme.nseg):
    #     sme.mask[i][sme.telluric[i] < 0.995] = sme.mask_values["bad"]

    sme.save(out_file)

    # sme.cscale_flag = "fix"
    # sme.wave = sme.wave[6:31]
    # sme.spec = sme.spec[6:31]
    # sme.synth = sme.synth[6:31]
    # sme.mask = sme.mask[6:31]
    # sme.telluric = sme.telluric[6:31]
    # save_as_idl(sme, "cnc55.inp")

    for fp in fitparameters:
        sme = solve(sme, fp, segments=np.arange(6, 31))
        fname = f"{target}_mask_new_out_{'_'.join(fp)}"
        out_file = os.path.join(examples_dir, "results", fname + ".sme")
        sme.save(out_file)

        plot_file = os.path.join(examples_dir, "results", fname + ".html")
        fig = plot_plotly.FinalPlot(sme)
        fig.save(filename=plot_file)
    # print(sme.citation())

    # Save results
    sme.save(out_file)

    # Plot results
    fig = plot_plotly.FinalPlot(sme)
    fig.save(filename=plot_file)
    print(f"Finished: {target}")
