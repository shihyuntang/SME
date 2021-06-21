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
from tqdm import tqdm


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


if __name__ == "__main__":
    # Define the location of all your files
    # this will put everything into the example dir
    target = "HD_22049"
    sdb = StellarDB()
    # star = sdb.auto_fill(target)
    star = sdb.load(target)
    alias = [re.sub(r"[-_ ]", "", s).lower() for s in star["id"]]

    examples_dir = dirname(realpath(__file__))
    data_dir = join(examples_dir, "data")

    # Find the correct data file for this target
    files = [fname for fname in os.listdir(data_dir) if fname.endswith(".fits")]
    isFound = False
    for fname in files:
        hdu = fits.open(join(data_dir, fname))
        header = hdu[0].header
        obj = header["OBJECT"]
        obj = re.sub(r"[-_ ]", "", obj).lower()
        hdu.close()
        if obj in alias:
            isFound = True
            break

    if not isFound:
        raise ValueError("No data file found")

    in_file = os.path.join(data_dir, fname)
    # in_file = os.path.join(examples_dir, f"results/{target}_mask_new.sme")

    vald_file = os.path.join(examples_dir, f"data/harps.lin")

    out_file_1 = os.path.join(examples_dir, f"results/{target}_mask_v2_1.sme")
    out_file_2 = os.path.join(examples_dir, f"results/{target}_mask_v2_2.sme")

    plot_file_1 = os.path.join(examples_dir, f"results/{target}_mask_v2_1.html")
    plot_file_2 = os.path.join(examples_dir, f"results/{target}_mask_v2_2.html")

    date_string = datetime.datetime.now().isoformat().replace(":", ".")
    log_file = os.path.join(examples_dir, f"results/{target}_{date_string}.log")

    # Start the logging to the file
    util.start_logging(log_file)

    # Load data from fits file
    hdu = fits.open(in_file)
    wave = hdu[1].data["WAVE"][0]
    flux = hdu[1].data["FLUX"][0]

    # Normalize using the maximum
    # This is important for the residuals later
    flux /= np.nanpercentile(flux, 95)

    # Get first guess from upper envelope
    _, high_idx = hl_envelopes_idx(flux, dmin=400, dmax=500)
    high_idx = np.array([0, *high_idx])
    cont = interp1d(
        wave[high_idx], flux[high_idx], kind="linear", fill_value="extrapolate"
    )(wave)

    # from scipy.interpolate import BSpline
    # from scipy.optimize import curve_fit

    # wave_high = []
    # flux_high = []
    # step = 500
    # for i in range(len(flux) // step + 1):
    #     w = wave[i * step : i * step + step]
    #     f = flux[i * step : i * step + step]
    #     mask = f > np.nanpercentile(f, 95)
    #     wave_high += [w[mask]]
    #     flux_high += [f[mask]]
    # wave_high = np.concatenate(wave_high)
    # flux_high = np.concatenate(flux_high)

    # wmin, wmax = wave[0], wave[-1]
    # n = 20
    # k = 3
    # t = np.linspace(wmin, wmax, n + k + 1)
    # xd = wave_high
    # func = lambda x, *c: BSpline(t, c, k, extrapolate=True)(x)
    # popt, pcov = curve_fit(
    #     func, xd, flux_high, p0=np.ones(n), loss="soft_l1", method="trf"
    # )
    # cont = BSpline(t, popt, k, extrapolate=True)(wave)

    # plt.plot(wave, flux)
    # plt.plot(wave, cont)
    # plt.plot(wave[high_idx], flux[high_idx], "+")
    # plt.show()

    flux /= cont

    # Get tellurics from Tapas
    ftapas = join(examples_dir, "data/tapas.ipac")
    dtapas = np.genfromtxt(ftapas, comments="\\", skip_header=36)
    wtapas, ftapas = dtapas[:, 0], dtapas[:, 1]
    # convert to angstrom
    wtapas *= 10
    # Normalize
    ftapas -= ftapas.min()
    ftapas /= ftapas.max()
    wtapas = wtapas[::-1]
    ftapas = ftapas[::-1]

    # Get first guess from upper envelope
    _, high_idx = hl_envelopes_idx(ftapas, dmin=100, dmax=100)
    high_idx = np.array([0, *high_idx])
    # Then fit the envelope, by limiting the residuals
    ctapas = interp1d(
        wtapas[high_idx], ftapas[high_idx], kind="cubic", fill_value="extrapolate"
    )(wtapas)

    # plt.plot(wtapas, ftapas)
    # plt.plot(wtapas, ctapas)
    # plt.plot(wtapas[high_idx], ftapas[high_idx], "+")
    # plt.show()
    ftapas /= ctapas

    # Transform to earth restframe
    # TODO: make this robust, so it works automatically
    # sme = SME.SME_Structure(wave=[wave], sob=[flux])
    # sme.vrad_flag = "each"
    # sme.cscale_flag = "none"
    # rv = determine_radial_velocity(sme, 0, [1], wtapas, ftapas, rv_bounds=(-200, 200))
    rv = -103
    c_light = const.c.to_value("km/s")
    rv_factor = np.sqrt((1 - rv / c_light) / (1 + rv / c_light))
    wave = rv_factor * wave

    sme = SME.SME_Structure(wave=[wave], sob=[flux])
    sme.mask = np.where(sme.wave > 6866, 1, 0)
    sme.vrad_flag = "each"
    sme.cscale_flag = "none"
    rv = determine_radial_velocity(sme, 0, [1], wtapas, ftapas)

    rv_factor = np.sqrt((1 - rv / c_light) / (1 + rv / c_light))
    wave = rv_factor * wave
    ftapas = np.interp(wave, wtapas, ftapas)
    wtapas = wave

    # This wavelength region consists of just telluric lines, and nothing else
    # This wavelength is in the earth reference frame
    mtapas = wtapas > 6866
    func = lambda x: gaussian_filter1d(ftapas[mtapas], x[0]) - flux[mtapas]
    res = least_squares(func, x0=[1], loss="soft_l1", method="trf")
    ftapas = gaussian_filter1d(ftapas, res.x[0])

    # plt.plot(wave, flux)
    # plt.plot(wtapas, ftapas)
    # plt.show()

    # Split the spectrum into arbitrary chunks
    # (This makes the progress bar more useful)
    # TODO: is this the same for all stars?
    nsteps = 10000
    wave = [
        wave[nsteps * i : nsteps * (i + 1)]
        for i in range(int(np.ceil(len(wave) / nsteps)))
    ]
    flux = [
        flux[nsteps * i : nsteps * (i + 1)]
        for i in range(int(np.ceil(len(flux) / nsteps)))
    ]
    ftapas = [
        ftapas[nsteps * i : nsteps * (i + 1)]
        for i in range(int(np.ceil(len(ftapas) / nsteps)))
    ]

    # err = hdu[1].data["ERR"]
    sme = SME.SME_Structure(wave=wave, sob=flux)
    sme.mu = np.geomspace(0.1, 1, num=7)
    sme.uncs = [1 / np.sqrt(spec) ** 2 for spec in sme.spec]
    # sme.mask = get_mask_from_neural_network(sme)
    sme.mask = sme.mask_values["line"]
    for i in range(sme.nseg):
        smoothed = gaussian_filter1d(sme.spec[i], 10)
        dsmoothed = np.gradient(smoothed)
        ddsmoothed = np.gradient(dsmoothed)
        # idx = np.abs(ddsmoothed) < np.nanpercentile(np.abs(ddsmoothed), 10)
        idx = np.abs(dsmoothed) < np.nanpercentile(np.abs(dsmoothed), 10)
        idx &= ddsmoothed < 0
        idx &= smoothed > np.median(smoothed)
        sme.mask[i][idx] = sme.mask_values["continuum"]
        sme.mask[i][sme.spec[i] == 0] = sme.mask_values["bad"]
        sme.mask[i][ftapas[i] < 0.995] = sme.mask_values["bad"]

        # idx = sme.mask[i] == sme.mask_values["continuum"]
        # plt.plot(sme.wave[i], sme.spec[i])
        # plt.plot(sme.wave[i], smoothed)
        # plt.plot(sme.wave[i][idx], sme.spec[i][idx], "+")
        # plt.show()

    # Add telluric data (without rayleigh scattering)
    sme.telluric = Iliffe_vector(values=ftapas)

    # Get first guess from literature values
    sme.teff = star.get("t_eff", 5065 * u.K).to_value("K")
    sme.logg = star.get("logg", 4.61 * u.one).to_value(1)
    monh = star.get("metallicity", -0.05 * u.one).to_value(1)
    sme.abund = Abund(monh, "asplund2009")
    sme.vmic = star.get("velocity_turbulence", 1 * u.km / u.s).to_value("km/s")
    # Test this
    sme.vmac = 2
    sme.vsini = 2.4

    # load the linelist
    sme.linelist = ValdFile(vald_file)

    # Set the atmosphere grid
    sme.atmo.source = "marcs2014.sav"
    sme.atmo.geom = "PP"
    sme.atmo.depth = "RHOX"
    sme.atmo.interp = "TAU"

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

    # Barycentric correction
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
    sme.vrad = -66 + 103

    sme = synthesize_spectrum(sme, segments="all")
    sme.save(out_file_1)

    # for i in range(sme.nseg):
    #     smoothed = gaussian_filter1d(sme.spec[i], 10)
    #     dsmoothed = np.gradient(smoothed)
    #     dsynth = np.gradient(sme.synth[i])
    #     resid = np.abs(dsmoothed - dsynth)
    #     mask = resid > np.nanpercentile(resid, 90)
    #     sme.mask[i][mask] = sme.mask_values["bad"]

    fig = plot_plotly.FinalPlot(sme)
    fig.save(filename=plot_file_1)

    # sme.vrad = (
    #     star["radial_velocity"].to_value("km/s") if "radial_velocity" in star else 0
    # )
    # sme.vrad -= correction
    # checked manually

    # Define any fitparameters you want
    # For abundances use: 'abund {El}', where El is the element (e.g. 'abund Fe')
    # For linelist use: 'linelist {Nr} {p}', where Nr is the number in the
    # linelist and p is the line parameter (e.g. 'linelist 17 gflog')
    # fitparameters = [
    #     ["monh"],
    #     ["teff"],
    #     ["logg", "vmic", "vmac", "vsini"],
    #     ["monh", "teff", "logg", "vmic", "vmac", "vsini"],
    # ]

    # # Restrict the linelist to relevant lines
    # # for this segment
    # rvel = 100
    # wmin, wmax = sme.wran[6][0], sme.wran[30][1]
    # wmin *= 1 - rvel / 3e5
    # wmax *= 1 + rvel / 3e5
    # sme.linelist = sme.linelist.trim(wmin, wmax)

    # # Start SME solver
    # # sme = synthesize_spectrum(sme, segments=np.arange(6, 31))
    # # sme.cscale_flag = "fix"
    # # sme.wave = sme.wave[6:31]
    # # sme.spec = sme.spec[6:31]
    # # sme.synth = sme.synth[6:31]
    # # sme.mask = sme.mask[6:31]
    # # sme.telluric = sme.telluric[6:31]
    # # save_as_idl(sme, "epseri.inp")

    # # sme.save(out_file)
    # for fp in fitparameters:
    #     sme = solve(sme, fp, segments=np.arange(6, 31))
    #     fname = f"{target}_mask_v2_{'_'.join(fp)}"
    #     out_file = os.path.join(examples_dir, "results", fname + ".sme")
    #     sme.save(out_file)

    #     plot_file = os.path.join(examples_dir, "results", fname + ".html")
    #     fig = plot_plotly.FinalPlot(sme)
    #     fig.save(filename=plot_file)

    # # print(sme.citation())

    # # Save results
    # sme.save(out_file_2)

    # Plot results
    # fig = plot_plotly.FinalPlot(sme)
    # fig.save(filename=plot_file_2)
    print(f"Finished: {target}")
