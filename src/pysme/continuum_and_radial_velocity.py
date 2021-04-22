"""
Determine continuum based on continuum mask
and fit best radial velocity to observation
"""

import logging
import warnings
from itertools import product

import emcee
import numpy as np
from scipy.constants import speed_of_light
from scipy.linalg import lu_factor, lu_solve
from scipy.ndimage.filters import median_filter
from scipy.optimize import least_squares, minimize_scalar, curve_fit
from scipy.signal import correlate, find_peaks
from tqdm import tqdm

from . import util
from .iliffe_vector import Iliffe_vector
from .sme_synth import SME_DLL

logger = logging.getLogger(__name__)

c_light = speed_of_light * 1e-3  # speed of light in km/s


def determine_continuum(sme, segment):
    """
    Fit a polynomial to the spectrum points marked as continuum
    The degree of the polynomial fit is determined by sme.cscale_flag

    Parameters
    ----------
    sme : SME_Struct
        input sme structure with sme.sob, sme.wave, and sme.mask
    segment : int
        index of the wavelength segment to use, or -1 when dealing with the whole spectrum

    Returns
    -------
    cscale : array of size (ndeg + 1,)
        polynomial coefficients of the continuum fit, in numpy order, i.e. largest exponent first
    """

    if segment < 0:
        return sme.cscale

    if "spec" not in sme or "wave" not in sme:
        # If there is no observation, we have no continuum scale
        warnings.warn("Missing data for continuum fit")
        cscale = [1]
    elif sme.cscale_flag in ["none", -3]:
        cscale = [1]
    elif sme.cscale_flag in ["fix", -1, -2]:
        # Continuum flag is set to no continuum
        cscale = sme.cscale[segment]
    else:
        # fit a line to the continuum points
        ndeg = sme.cscale_degree

        # Extract points in this segment
        x, y = sme.wave, sme.spec
        if "mask" in sme:
            m = sme.mask
        else:
            m = sme.spec.copy()
            m[:] = sme.mask_value["line"]

        if "uncs" in sme:
            u = sme.uncs
        else:
            u = sme.spec.copy()
            u[:] = 1
        x, y, m, u = x[segment], y[segment], m[segment], u[segment]

        # Set continuum mask
        if np.all(m != sme.mask_values["continuum"]):
            # If no continuum mask has been set
            # Use the effective wavelength ranges of the lines to determine continuum points
            logger.info(
                "No Continuum mask was set in segment %s, "
                "Using effective wavelength range of lines to find continuum instead",
                segment,
            )
            cont = get_continuum_mask(x, y, sme.linelist, mask=m)
            # Save mask for next iteration
            m[cont == 2] = sme.mask_values["continuum"]
            logger.debug("Continuum mask points: %i", np.count_nonzero(cont == 2))

        cont = m == sme.mask_values["continuum"]
        x = x - x[0]
        x, y, u = x[cont], y[cont], u[cont]

        # Fit polynomial
        try:
            func = lambda coef: (np.polyval(coef, x) - y) / u
            c0 = np.polyfit(x, y, deg=ndeg)
            res = least_squares(func, x0=c0)
            cscale = res.x
        except TypeError:
            warnings.warn("Could not fit continuum, set continuum mask?")
            cscale = [1]

    return cscale


def get_continuum_mask(wave, synth, linelist, threshold=0.1, mask=None):
    """
    Use the effective wavelength range of the lines,
    to find wavelength points that should be unaffected by lines
    However one usually has to ignore the weak lines, as most points are affected by one line or another
    Therefore keep increasing the threshold until enough lines have been found (>10%)

    Parameters
    ----------
    wave : array of size (n,)
        wavelength points
    linelist : LineList
        LineList object that was input into the Radiative Transfer
    threshold : float, optional
        starting threshold, lines with depth below this value are ignored
        the actual threshold is increased until enough points are found (default: 0.1)

    Returns
    -------
    mask : array(bool) of size (n,)
        True for points between lines and False for points within lines
    """

    if "depth" not in linelist.columns:
        raise ValueError(
            "No depth specified in the linelist, can't auto compute the mask"
        )

    if threshold <= 0:
        threshold = 0.01

    if mask is None:
        mask = np.full(len(wave), 1)

    # TODO make this better
    dll = SME_DLL()
    dll.linelist = linelist

    width = dll.GetLineRange()
    # TODO: optimize this
    temp = False
    while np.count_nonzero(temp) < len(wave) * 0.1:
        temp = np.full(len(wave), True)
        for i, line in enumerate(width):
            if linelist["depth"][i] > threshold:
                w = (wave >= line[0]) & (wave <= line[1])
                temp[w] = False

        # TODO: Good value to increase threshold by?
        temp[mask == 0] = False
        threshold *= 1.1

    mask[temp] = 2

    logger.debug("Ignoring lines with depth < %f", threshold)
    return mask


def determine_radial_velocity(sme, segment, cscale, x_syn, y_syn):
    """
    Calculate radial velocity by using cross correlation and
    least-squares between observation and synthetic spectrum

    Parameters
    ----------
    sme : SME_Struct
        sme structure with observed spectrum and flags
    segment : int
        which wavelength segment to handle, -1 if its using the whole spectrum
    cscale : array of size (ndeg,)
        continuum coefficients, as determined by e.g. determine_continuum
    x_syn : array of size (n,)
        wavelength of the synthetic spectrum
    y_syn : array of size (n,)
        intensity of the synthetic spectrum

    Raises
    ------
    ValueError
        if sme.vrad_flag is not recognized

    Returns
    -------
    rvel : float
        best fit radial velocity for this segment/whole spectrum
        or None if no observation is present
    """

    if "spec" not in sme or "wave" not in sme:
        # No observation no radial velocity
        warnings.warn("Missing data for radial velocity determination")
        rvel = 0
    elif sme.vrad_flag == "none":
        # vrad_flag says don't determine radial velocity
        rvel = sme.vrad[segment]
    elif sme.vrad_flag == "whole" and np.size(segment) == 1:
        # We are inside a segment, but only want to determine rv at the end
        rvel = 0
    elif sme.vrad_flag == "fix":
        rvel = sme.vrad[segment]
    else:
        # Fit radial velocity
        # Extract data
        x, y = sme.wave, sme.spec
        if "mask" in sme:
            m = sme.mask
        else:
            m = sme.spec.copy()
            m[:] = sme.mask_value["line"]

        if "uncs" in sme:
            u = sme.uncs
        else:
            u = sme.spec.copy()
            u[:] = 1

        # Only this one segment
        x_obs = x[segment]
        y_obs = y[segment].copy()
        u_obs = u[segment]
        mask = m[segment]

        if sme.vrad_flag == "each":
            # apply continuum
            if cscale is not None:
                cont = np.polyval(cscale, x_obs - x_obs[0])
            else:
                warnings.warn(
                    "No continuum scale passed to radial velocity determination"
                )
                cont = np.ones_like(y_obs)
            y_obs /= cont
        elif sme.vrad_flag == "whole":
            # All segments
            if cscale is not None:
                cscale = np.atleast_2d(cscale)
                cont = [
                    np.polyval(c, x_obs[i] - x_obs[i][0]) for i, c in enumerate(cscale)
                ]
            else:
                warnings.warn(
                    "No continuum scale passed to radial velocity determination"
                )
                cont = [1 for _ in range(len(x_obs))]

            for i in range(len(y_obs)):
                y_obs[i] /= cont[i]

            x_obs = x_obs.ravel()
            y_obs = y_obs.ravel()
            u_obs = u_obs.ravel()
            x_syn = np.concatenate(x_syn)
            y_syn = np.concatenate(y_syn)
            mask = mask.ravel()
        else:
            raise ValueError(
                f"Radial velocity flag {sme.vrad_flag} not recognised, expected one of 'each', 'whole', 'none'"
            )

        mask = mask == sme.mask_values["line"]
        x_obs = x_obs[mask]
        y_obs = y_obs[mask]
        u_obs = u_obs[mask]
        y_tmp = np.interp(x_obs, x_syn, y_syn)

        rv_bounds = (-100, 100)
        if np.all(sme.vrad[segment] == 0):
            # Get a first rough estimate from cross correlation
            # Subtract continuum level of 1, for better correlation
            corr = correlate(
                y_obs - np.median(y_obs), y_tmp - np.median(y_tmp), mode="same"
            )
            offset = np.argmax(corr)

            x1 = x_obs[offset]
            x2 = x_obs[len(x_obs) // 2]
            rvel = c_light * (1 - x2 / x1)

            rvel = np.clip(rvel, *rv_bounds)
        else:
            if sme.vrad_flag == "whole":
                rvel = sme.vrad[0]
            else:
                rvel = sme.vrad[segment]

        # Then minimize the least squares for a better fit
        # as cross correlation can only find
        def func(rv):
            rv_factor = np.sqrt((1 - rv / c_light) / (1 + rv / c_light))
            shifted = interpolator(x_obs * rv_factor)
            # shifted = np.interp(x_obs[lines], x_syn * rv_factor, y_syn)
            resid = (y_obs - shifted) / u_obs
            resid = np.nan_to_num(resid, copy=False)
            return resid

        interpolator = lambda x: np.interp(x, x_syn, y_syn)
        res = least_squares(func, x0=rvel, loss="soft_l1", bounds=rv_bounds)
        rvel = res.x[0]

    return rvel


def null_result(nseg, ndeg=0):
    vrad, vrad_unc = np.zeros(nseg), np.zeros((nseg, 2))
    cscale, cscale_unc = np.zeros((nseg, ndeg + 1)), np.zeros((nseg, ndeg + 1, 2))
    cscale[:, -1] = 1
    return vrad, vrad_unc, cscale, cscale_unc


def determine_rv_and_cont(sme, segment, x_syn, y_syn):
    """
    Fits both radial velocity and continuum level simultaneously
    by comparing the synthetic spectrum to the observation

    The best fit is determined using robust least squares between
    a shifted and scaled synthetic spectrum and the observation

    Parameters
    ----------
    sme : SME_Struct
        contains the observation
    segment : int
        wavelength segment to fit
    x_syn : array of size (ngrid,)
        wavelength of the synthetic spectrum
    y_syn : array of size (ngrid,)
        intensity of the synthetic spectrum

    Returns
    -------
    vrad : float
        radial velocity in km/s
    vrad_unc : float
        radial velocity uncertainty in km/s
    cscale : array of size (ndeg+1,)
        polynomial coefficients of the continuum
    cscale_unc : array if size (ndeg + 1,)
        uncertainties of the continuum coefficients
    """

    if np.isscalar(segment):
        segment = [segment]
    nseg = len(segment)

    if sme.cscale_flag in ["none", "fix"] and sme.vrad_flag in ["none", "fix"]:
        vrad, vunc, cscale, cunc = null_result(nseg, sme.cscale_degree)
        if sme.vrad_flag == "fix":
            vrad = sme.vrad[segment]
        if sme.cscale_flag == "fix":
            cscale = sme.cscale[segment]
        return vrad, vunc, cscale, cunc

    if "spec" not in sme or "wave" not in sme:
        # No observation no radial velocity
        logger.warning("Missing data for radial velocity/continuum determination")
        return null_result(nseg, sme.cscale_degree)

    if "mask" not in sme:
        sme.mask = np.full(sme.spec.size, sme.mask_values["line"])
    if "uncs" not in sme:
        sme.uncs = np.full(sme.spec.size, 1.0)

    if np.all(sme.mask_bad[segment].ravel()):
        warnings.warn(
            "Only bad pixels in this segments, can't determine radial velocity/continuum",
            UserWarning,
        )
        return null_result(nseg, sme.cscale_degree)

    if x_syn.ndim == 1:
        x_syn = x_syn[None, :]
    if y_syn.ndim == 1:
        y_syn = y_syn[None, :]

    if x_syn.shape[0] != nseg or y_syn.shape[0] != nseg:
        raise ValueError(
            "Size of synthetic spectrum, does not match the number of requested segments"
        )

    mask = sme.mask_good[segment]
    x_obs = sme.wave[segment][mask]
    y_obs = sme.spec[segment][mask]
    x_num = x_obs - sme.wave[segment][:, 0]

    if x_obs.size <= sme.cscale_degree:
        warnings.warn("Not enough good pixels to determine radial velocity/continuum")
        return null_result(nseg)

    if sme.cscale_flag in [-3, "none"]:
        cflag = False
        cscale = np.ones((nseg, 1))
        ndeg = 0
    elif sme.cscale_flag in [-1, -2, "fix"]:
        cflag = False
        cscale = sme.cscale[segment]
        ndeg = cscale.shape[1] - 1
    elif sme.cscale_flag in [0, "constant"]:
        ndeg = 0
        cflag = True
    elif sme.cscale_flag in [1, "linear"]:
        ndeg = 1
        cflag = True
    elif sme.cscale_flag in [2, "quadratic"]:
        ndeg = 2
        cflag = True
    else:
        raise ValueError("cscale_flag not recognized")

    if cflag:
        if sme.cscale is not None:
            cscale = sme.cscale[segment]
        else:
            cscale = np.zeros(nseg, ndeg + 1)
            for i, seg in enumerate(segment):
                cscale[i, -1] = np.nanpercentile(y_obs[seg], 95)

    # Even when the vrad_flag is set to whole
    # you still want to fit the rv of the individual segments
    # just for the continuum fit
    if sme.vrad_flag == "none":
        vrad = np.zeros(len(segment))
        vflag = False
    elif sme.vrad_flag == "whole":
        vrad = sme.vrad[:1]
        vflag = True
    elif sme.vrad_flag == "each":
        vrad = sme.vrad[segment]
        vflag = True
    elif sme.vrad_flag == "fix":
        vrad = sme.vrad[segment]
        vflag = False
    else:
        raise ValueError(f"Radial velocity Flag not understood {sme.vrad_flag}")

    # Limit shift to half an order
    x1, x2 = x_obs[:, 0], x_obs[:, [s // 4 for s in x_obs.shape[1]]]
    rv_limit = np.abs(c_light * (1 - x2 / x1))
    if sme.vrad_flag == "whole":
        rv_limit = np.min(rv_limit)

    # Use Cross corellatiom as a first guess for the radial velocity
    # This uses the median as the continuum
    if vrad is None:
        y_tmp = np.interp(x_obs, x_syn, y_syn, left=1, right=1)
        corr = correlate(y_obs - np.median(y_obs), y_tmp - 1, mode="same")
        offset = np.argmax(corr)
        x1, x2 = x_obs[offset], x_obs[len(x_obs) // 2]
        vrad = c_light * (1 - x2 / x1)
        if np.abs(vrad) >= rv_limit:
            logger.warning(
                "Radial Velocity could not be estimated from cross correlation, using initial guess of 0 km/h. Please check results!"
            )
            vrad = 0

    def log_prior(rv, cscale, nwalkers):
        prior = np.zeros(nwalkers)
        # Add the prior here
        # TODO reject too large/small rv values in a prior
        where = np.full(nwalkers, False)
        if vflag:
            where |= np.any(np.abs(rv) > rv_limit, axis=1)
        if cflag:
            where |= np.any(cscale[:, :, -1] < 0, axis=1)
            if ndeg == 1:
                where |= np.any(
                    (cscale[:, :, -1] + cscale[:, :, -2] * x_num[:, -1]) < 0, axis=1
                )
            elif ndeg == 2:
                for i in range(nseg):
                    where |= np.any(
                        cscale[:, i, None, -1]
                        + cscale[:, i, None, -2] * x_num[i]
                        + cscale[:, i, None, -3] * x_num[i] ** 2
                        < 0,
                        axis=1,
                    )
        prior[where] = -np.inf
        return prior

    def log_prob(par, sep, nseg, ndeg):
        """
        par : array of shape (nwalkers, ndim)
            ndim = 1 for radial velocity + continuum polynomial coeficients
        """
        nwalkers = par.shape[0]
        rv = par[:, :sep] if vflag else vrad[None, :]
        if rv.shape[0] == 1 and nwalkers > 1:
            rv = np.tile(rv, [nwalkers, 1])
        if rv.shape[1] == 1 and nseg > 1:
            rv = np.tile(rv, [1, nseg])
        if cflag:
            cs = par[:, sep:]
            cs.shape = nwalkers, nseg, ndeg + 1
        else:
            cs = cscale[None, ...]

        prior = log_prior(rv, cs, nwalkers)

        # Apply RV shift
        rv_factor = np.sqrt((1 - rv / c_light) / (1 + rv / c_light))
        total = np.zeros(nwalkers)
        for i in range(nseg):
            x = x_obs[i][None, :] * rv_factor[:, i, None]
            model = np.interp(x, x_syn[i], y_syn[i], left=0, right=0)

            # Apply continuum
            y = np.zeros_like(x_num[i])[None, :]
            for j in range(ndeg + 1):
                y = y * x_num[i] + cs[:, i, j, None]
            model *= y

            # Ignore the non-overlapping parts of the spectrum
            mask = model == 0
            npoints = mask.shape[1] - np.count_nonzero(mask, axis=1)
            resid = (model - y_obs[i]) ** 2
            resid[mask] = 0
            prob = -0.5 * np.sum(resid, axis=-1)
            # Need to rescale here, to account for the ignored points before
            prob *= mask.shape[1] / npoints
            prob[np.isnan(prob)] = -np.inf
            total += prob
        return prior + total

    sep = len(vrad) if vflag else 0
    ndim, p0, scale = 0, [], []
    if vflag:
        ndim += len(vrad)
        p0 += list(vrad)
        scale += [1] * len(vrad)
    if cflag:
        ndim += cscale.size
        p0 += list(cscale.ravel())
        scale += [0.001] * cscale.size
    p0 = np.array(p0)[None, :]
    scale = np.array(scale)[None, :]

    max_n = 10000
    ncheck = 100
    nburn = 300
    nwalkers = max(2 * ndim + 1, 10)
    p0 = p0 + np.random.randn(nwalkers, ndim) * scale
    # If the original guess is good then DEMove is much faster, and sometimes just as good
    # However StretchMove is much more robust to the initial starting value
    moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob, vectorize=True, moves=moves, args=(sep, nseg, ndeg)
    )
    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty((max_n // ncheck + 1, ndim))
    # This will be useful to testing convergence
    # old_tau = 0

    # Now we'll sample for up to max_n steps
    with tqdm(leave=False, desc="RV", total=max_n) as t:
        for _ in sampler.sample(p0, iterations=max_n):
            t.update()
            # Only check convergence every 100 steps
            if sampler.iteration < 2 * nburn or sampler.iteration % ncheck != 0:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0, discard=sampler.iteration - ncheck)
            autocorr[index] = tau
            index += 1

            # Check convergence
            converged = np.all(tau * 100 < sampler.iteration - nburn)
            # converged &= np.all(np.abs(old_tau - tau) < 0.01 * tau)
            # old_tau = tau
            if converged:
                break

    if sampler.iteration == max_n:
        logger.warning(
            "The radial velocity did not converge within the limit. Check the results!"
        )

    samples = sampler.get_chain(flat=True, discard=nburn)
    _, vrad_unc, _, cscale_unc = null_result(nseg, ndeg)
    if vflag:
        vmin, vrad, vmax = np.percentile(samples[:, :sep], (32, 50, 68), axis=0)
        vrad_unc[:, 0] = vrad - vmin
        vrad_unc[:, 1] = vmax - vrad

    if cflag:
        vmin, cscale, vmax = np.percentile(samples[:, sep:], (32, 50, 68), axis=0)
        vmin.shape = cscale.shape = vmax.shape = nseg, ndeg + 1

        cscale_unc[..., 0] = cscale - vmin
        cscale_unc[..., 1] = vmax - cscale

    if sme.vrad_flag == "whole":
        vrad = np.tile(vrad, [nseg])

    return vrad, vrad_unc, cscale, cscale_unc


def cont_fit(sme, segment, x_syn, y_syn, rvel=0):
    """
    Fit a continuum when no continuum points exist

    Parameters
    ----------
    sme : SME_Struct
        sme structure with observation data
    segment : int
        index of the wavelength segment to fit
    x_syn : array of size (n,)
        wavelengths of the synthetic spectrum
    y_syn : array of size (n,)
        intensity of the synthetic spectrum
    rvel : float, optional
        radial velocity in km/s to apply to the wavelength (default: 0)

    Returns
    -------
    continuum : array of size (ndeg,)
        continuum fit polynomial coefficients
    """

    if sme.cscale_flag == "none":
        return [1]
    elif sme.cscale_flag == "fix":
        return sme.cscale[segment]
    # else
    x = sme.wave[segment]
    y = sme.spec[segment]
    m = sme.mask_good[segment]

    rv_factor = np.sqrt((1 - rvel / c_light) / (1 + rvel / c_light))
    xp = x * rv_factor
    yp = np.interp(xp, x_syn, y_syn)

    xs = x - x[0]
    xs, y, yp = xs[m], y[m], yp[m]

    func = lambda x, *c: yp * np.polyval(c, xs)

    deg = sme.cscale_degree
    p0 = np.zeros(deg + 1)
    p0[-1] = np.nanmedian(y)
    popt, pcov = curve_fit(func, xs, y, p0=p0)

    # For debugging
    # plt.plot(x, yp * np.polyval(popt, xs), label="model")
    # plt.plot(x, y, label="observation")
    # plt.show()

    return popt[None, :]


def match_rv_continuum(sme, segments, x_syn, y_syn):
    """
    Match both the continuum and the radial velocity of observed/synthetic spectrum

    Note that the parameterization of the continuum is different to old SME !!!

    Parameters
    ----------
    sme : SME_Struct
        input sme structure with all the parameters
    segment : int
        index of the wavelength segment to match, or -1 when dealing with the whole spectrum
    x_syn : array of size (n,)
        wavelength of the synthetic spectrum
    y_syn : array of size (n,)
        intensitz of the synthetic spectrum

    Returns
    -------
    rvel : float
        new radial velocity
    cscale : array of size (ndeg + 1,)
        new continuum coefficients
    """

    vrad, vrad_unc, cscale, cscale_unc = null_result(sme.nseg, sme.cscale_degree)
    if sme.cscale_flag == "none" and sme.vrad_flag == "none":
        return cscale, cscale_unc, vrad, vrad_unc

    if np.isscalar(segments):
        segments = [segments]

    if sme.cscale_type in ["whole", "mcmc"]:
        if sme.vrad_flag in ["each", "none", "fix"]:
            for s in tqdm(segments, desc="RV/Cont", leave=False):
                vrad[s], vrad_unc[s], cscale[s], cscale_unc[s] = determine_rv_and_cont(
                    sme, s, x_syn[s], y_syn[s]
                )
        elif sme.vrad_flag == "whole":
            wave = Iliffe_vector(values=[x_syn[s] for s in segments])
            smod = Iliffe_vector(values=[y_syn[s] for s in segments])
            s = segments
            vrad[s], vrad_unc[s], cscale[s], cscale_unc[s] = determine_rv_and_cont(
                sme, s, wave, smod
            )
        else:
            raise ValueError(f"Radial velocity flag {sme.vrad_flag} not understood")
    elif sme.cscale_type == "mask":
        for s in segments:
            cscale[s] = determine_continuum(sme, s)
            vrad[s] = determine_radial_velocity(sme, s, cscale[s], x_syn[s], y_syn[s])

        if sme.vrad_flag == "whole":
            s = segments
            vrad[s] = determine_radial_velocity(
                sme, s, cscale[s], [x_syn[s] for s in s], [y_syn[s] for s in s]
            )
    elif sme.cscale_type == "match":
        for s in segments:
            cscale[s][-1] = np.nanpercentile(sme.spec[s], 95)
            vrad[s] = determine_radial_velocity(sme, s, cscale[s], x_syn[s], y_syn[s])
            cscale[s] = cont_fit(sme, s, x_syn[s], y_syn[s], rvel=vrad[s])

        if sme.vrad_flag == "whole":
            s = segments
            vrad[s] = determine_radial_velocity(
                sme, s, cscale[s], [x_syn[s] for s in s], [y_syn[s] for s in s]
            )
    else:
        raise ValueError(
            f"Did not understand cscale_type, expected one of ('whole', 'mask'), but got {sme.cscale_type}."
        )

    # Keep values from unused segments
    select = np.arange(sme.nseg)
    mask = np.full(select.shape, True)
    for seg in segments:
        mask &= select != seg
    vrad[mask] = sme.vrad[mask]
    cscale[mask] = sme.cscale[mask]

    return cscale, cscale_unc, vrad, vrad_unc
