""" Minimum working example of an SME script
"""
import datetime
import os
import os.path
import re
import sys
from os.path import basename, dirname, exists, join, realpath

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy.io import fits
from data_sources.StellarDB import StellarDB
from flex.flex import FlexFile
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d, median_filter
from scipy.optimize import least_squares
from tqdm import tqdm

from pysme import sme as SME
from pysme import util
from pysme.abund import Abund
from pysme.continuum_and_radial_velocity import determine_radial_velocity
from pysme.gui import plot_plotly
from pysme.iliffe_vector import Iliffe_vector
from pysme.linelist.vald import ValdFile
from pysme.solve import solve


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


def load_star(target, update=True):
    sdb = StellarDB(sources=["simbad", "exoplanets_nasa"])
    if update:
        star = sdb.auto_fill(target)
    star = sdb.load(target)
    return star


def get_value(star, name, unit, alt=None):
    if name in star:
        return star[name].to_value(unit)
    else:
        return alt


def load_fname(star, data_dir):
    alias = [re.sub(r"[-_ *]", "", s).lower() for s in star["id"]]

    # Find the correct data file for this target
    files = [fname for fname in os.listdir(data_dir) if fname.endswith(".fits")]
    isFound = False
    for fname in files:
        hdu = fits.open(join(data_dir, fname))
        header = hdu[0].header
        obj = header["OBJECT"]
        obj = re.sub(r"[-_ *]", "", obj).lower()
        hdu.close()
        if obj in alias:
            isFound = True
            break
    if not isFound:
        raise ValueError("No data file found")
    return join(data_dir, fname)


def load_tellurics():
    # Get tellurics from Tapas
    examples_dir = dirname(realpath(__file__))
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
    return wtapas, ftapas


def normalize(wave, flux, dmin=400, dmax=500):
    flux /= np.nanpercentile(flux, 95)
    # Get first guess from upper envelope
    mflux = median_filter(flux, 3)
    _, high_idx = hl_envelopes_idx(mflux, dmin=dmin, dmax=dmax)
    high_idx = np.array([0, *high_idx])
    cont = interp1d(
        wave[high_idx], flux[high_idx], kind="linear", fill_value="extrapolate"
    )(wave)
    flux /= cont
    return flux


def vel_shift(rv, wave):
    c_light = const.c.to_value("km/s")
    rv_factor = np.sqrt((1 - rv / c_light) / (1 + rv / c_light))
    wave = rv_factor * wave
    return wave


def get_rv_tell(wave, flux, wtell, ftell):
    sme = SME.SME_Structure(wave=[wave], sob=[flux])
    sme.mask = np.where(sme.wave > 6866, 1, 0)
    sme.vrad_flag = "each"
    sme.cscale_flag = "none"
    rv_tell = determine_radial_velocity(sme, wtell, ftell, 0, rv_bounds=(-200, 200))
    # plt.plot(vel_shift(rv, wave[sme.mask[0] == 1]), flux[sme.mask[0] == 1])
    # plt.plot(wtapas[wtapas > 6866], ftapas[wtapas > 6866])
    # plt.show()
    return rv_tell


def match_tell_with_obs(wave, flux, wtell, ftell):
    ftell = np.interp(wave, wtell, ftell)
    wtell = wave
    mtell = wtell > 6866
    func = lambda x: gaussian_filter1d(ftell[mtell], x[0]) - flux[mtell]
    res = least_squares(func, x0=[1], loss="soft_l1", method="trf")
    ftell = gaussian_filter1d(ftell, res.x[0])
    # plt.plot(wave, flux)
    # plt.plot(wtell, ftell)
    # plt.show()
    return wtell, ftell


def split_into_segments(wave, flux, ftell):
    nsteps = 10000
    wave = [
        wave[nsteps * i : nsteps * (i + 1)]
        for i in range(int(np.ceil(len(wave) / nsteps)))
    ]
    flux = [
        flux[nsteps * i : nsteps * (i + 1)]
        for i in range(int(np.ceil(len(flux) / nsteps)))
    ]
    ftell = [
        ftell[nsteps * i : nsteps * (i + 1)]
        for i in range(int(np.ceil(len(ftell) / nsteps)))
    ]
    return wave, flux, ftell


def run(target, mask=None, linelist="harps.lin", segments="all"):
    examples_dir = dirname(realpath(__file__))
    data_dir = join(examples_dir, "data")
    # /home/ansgar/Documents/Python/sme/examples/paper/data/ADP.2019-11-16T01 15 37.789.fits
    # Define the location of all your files
    # this will put everything into the example dir
    # Get stellar data from online sources
    star = load_star(target, update=False)
    # Find the correct data file for this target
    in_file = load_fname(star, data_dir)
    vald_file = join(examples_dir, f"data/{linelist}")
    mid_file = join(examples_dir, f"results/{target}_inp.sme")
    plot_file = join(examples_dir, f"results/{target}.html")

    # Load data from fits file
    hdu = fits.open(in_file)
    wave = hdu[1].data["WAVE"][0]
    flux = hdu[1].data["FLUX"][0]
    if mask is not None:
        for m in mask:
            flux[m[0] : m[1]] = 0
    flux = normalize(wave, flux)

    # Get tellurics from Tapas
    wtapas, ftapas = load_tellurics()
    ftapas = normalize(wtapas, ftapas, dmin=100, dmax=100)

    # Transform to earth restframe
    rv_tell = get_rv_tell(wave, flux, wtapas, ftapas)
    wave = vel_shift(rv_tell, wave)

    # Match the broadening of the observation
    wtapas, ftapas = match_tell_with_obs(wave, flux, wtapas, ftapas)

    # Split the spectrum into arbitrary chunks
    # (This makes the progress bar more useful)
    # TODO: is this the same for all stars?
    wave, flux, ftapas = split_into_segments(wave, flux, ftapas)

    # Create the SME structure
    sme = SME.SME_Structure(wave=wave, sob=flux)
    sme.meta["object"] = target
    sme.mu = np.geomspace(0.1, 1, num=7)
    # Use simple shot noise assumption for uncertainties
    sme.uncs = [np.sqrt(spec) ** 2 for spec in sme.spec]
    # Add telluric data (without rayleigh scattering)
    sme.telluric = Iliffe_vector(values=ftapas)
    # Create first mask by removing the telluric offset
    sme.mask = sme.mask_values["line"]
    for i in range(sme.nseg):
        sme.mask[i][sme.telluric[i] < 0.995] = sme.mask_values["bad"]
        sme.mask[i][sme.spec[i] == 0] = sme.mask_values["bad"]

    # Get first guess from literature values
    sme.teff = get_value(star, "t_eff", "K")
    sme.logg = get_value(star, "logg", 1)
    monh = get_value(star, "metallicity", 1, 0)
    sme.abund = Abund(monh, "asplund2009")
    # There is no reference for these
    sme.vmic = 1  # get_value(star, "velocity_turbulence", "km/s", 1)
    sme.vmac = 2
    sme.vsini = get_value(star, "velocity_rotation", "km/s", 2)

    # Load the linelist
    # and restrict the linelist to relevant lines
    sme.linelist = ValdFile(vald_file)
    wmin = sme.wran[segments[0]][0]
    wmax = sme.wran[segments[-1]][-1]
    sme.linelist = sme.linelist.trim(wmin, wmax, rvel=100)

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

    # Set radial velocity and continuum settings
    # Set RV and Continuum flags
    sme.vrad_flag = "whole"
    sme.cscale_flag = "linear"
    sme.cscale_type = "match"
    sme.vrad = 0

    # Determine the radial velocity offsets
    # sme = synthesize_spectrum(sme, segments=segments)

    # Save and Plot
    sme.save(mid_file)
    fig = plot_plotly.FinalPlot(sme)
    fig.save(filename=plot_file)

    return sme


def run_again(target, segments="all"):
    examples_dir = dirname(realpath(__file__))
    mid_mask_file = join(examples_dir, f"results/{target}_inp_mask.sme")
    mid_file = join(examples_dir, f"results/{target}_inp.sme")

    mask_file = join(
        examples_dir,
        "results_spline/HD_22049_mask_new_out_monh_teff_logg_vmic_vmac_vsini.sme",
    )

    try:
        # ff = FlexFile.read(mid_mask_file)
        # ff.header["cscale_type"] = "match"
        # ff.header["cscale_flag"] = "linear"
        # ff.write(mid_mask_file)
        sme = SME.SME_Structure.load(mid_mask_file)
        mid_file = mid_mask_file
    except FileNotFoundError:
        # ff = FlexFile.read(mid_file)
        # ff.header["cscale_type"] = "match"
        # ff.header["cscale_flag"] = "linear"
        # ff.write(mid_file)
        sme = SME.SME_Structure.load(mid_file)

    # # We fix broken save files
    # try:
    #     if isinstance(sme.cscale._values, Iliffe_vector):
    #         sme.cscale = [sme.cscale._values.data[str(i)] for i in range(sme.nseg)]
    #         sme.save(mid_file)
    #         sme = SME.SME_Structure.load(mid_file)
    # except:
    #     pass
    # sme.save(mid_file)
    # sme = SME.SME_Structure.load(mid_file)

    # Import manual mask
    # we needed to run sme once to get the correct radial velocities
    sme_mask = SME.SME_Structure.load(mask_file)
    sme = sme.import_mask(sme_mask, keep_bpm=True)

    # sme = synthesize_spectrum(sme, segments=segments)
    # Save and Plot
    sme.save(mid_file)
    # fig = plot_plotly.FinalPlot(sme)
    # fig.save(filename=plot_file)
    return sme


def fit(sme, segments="all"):
    examples_dir = dirname(realpath(__file__))
    # Define any fitparameters you want
    # For abundances use: 'abund {El}', where El is the element (e.g. 'abund Fe')
    # For linelist use: 'linelist {Nr} {p}', where Nr is the number in the
    # linelist and p is the line parameter (e.g. 'linelist 17 gflog')
    fitparameters = [["monh", "teff", "logg", "vmic", "vmac", "vsini"]]
    sme.cscale_type = "match"
    sme.cscale_flag = "linear"
    sme.vrad_flag = "whole"
    sme.vrad = None
    sme.cscale = None

    for fp in fitparameters:
        tmp = os.path.join(examples_dir, f"results/{target}.json")
        sme = solve(sme, fp, segments=segments, filename=tmp)
        fname = f"{target}_{'_'.join(fp)}"
        out_file = os.path.join(examples_dir, "results", fname + ".sme")
        sme.save(out_file)

        plot_file = os.path.join(examples_dir, "results", fname + ".html")
        fig = plot_plotly.FinalPlot(sme)
        fig.save(filename=plot_file, auto_open=False)

    # Save results
    out_file = join(examples_dir, f"results/{target}_out.sme")
    sme.save(out_file)
    print(sme.citation())

    # Plot results
    fig = plot_plotly.FinalPlot(sme)
    fig.save(filename=plot_file, auto_open=False)
    print(f"Finished: {target}")
    return sme


if __name__ == "__main__":
    targets = [
        "AU_Mic",
        "Eps_Eri",
        "HN_Peg",
        "HD_102195",
        "HD_130322",
        "HD_179949",
        "HD_189733",
        "55_Cnc",
        "WASP-18",
    ]
    rv = {
        "AU_Mic": {"star": -30, "tell": -56},
        "Eps_Eri": {"star": 66, "tell": -103},
        "HN_Peg": {"star": 0, "tell": -101},
        "HD_102195": {"star": 0, "tell": -100},
        "HD_130322": {"star": 0, "tell": -55},
        "HD_179949": {"star": 0, "tell": -103},
        "HD_189733": {"star": 0, "tell": -96},
        "55_Cnc": {"star": 0, "tell": -82},
        "WASP-18": {"star": 0, "tell": -98},
    }
    masked = {
        "AU_Mic": [
            (4982, 5527),
            (10561, 10962),
            (15063, 15287),
            (18559, 18925),
            (31888, 32002),
            (55757, 55892),
            (107806, 107983),
            (277889, 278216),
        ],
        "HN_Peg": [
            (128657, 128688),
            (191706, 191797),
            (216217, 216274),
            (230970, 231002),
            (284504, 284525),
        ],
        "HD_102195": [
            (172920, 172934),
            (204149, 204179),
            (217108, 217127),
            (246136, 246158),
            (270349, 270375),
            (275453, 275473),
            (284042, 284073),
            (300087, 300117),
        ],
        "HD_130322": [
            (99454, 99483),
            (155753, 155766),
            (167969, 167977),
            (174306, 174326),
            (186611, 186627),
            (195202, 195213),
            (201398, 201412),
            (227032, 227048),
            (244238, 244249),
            (244248, 244254),
            (250164, 250186),
            (250810, 250831),
            (282881, 282894),
            (283855, 283867),
            (293643, 293654),
            (295342, 295352),
        ],
        "HD_179949": [
            (67173, 67193),
            (87649, 87658),
            (186042, 186058),
            (223959, 223967),
            (228333, 228345),
            (272063, 272132),
            (276278, 276392),
        ],
        "HD_189733": [
            (103750, 103758),
            (107170, 107178),
            (110191, 110206),
            (111987, 112007),
            (186224, 186238),
            (206396, 206449),
            (234368, 234381),
            (236937, 236972),
            (249979, 249993),
            (256079, 256085),
            (263106, 263168),
        ],
        "55_Cnc": [(198466, 198469), (272134, 272144), (297364, 297371)],
        "WASP-18": [
            (21199, 21219),
            (57432, 57450),
            (138968, 138980),
            (151274, 151288),
            (156029, 156041),
            (168338, 168355),
            (196785, 196813),
            (200006, 200040),
            (205810, 205832),
            (208697, 208751),
            (219797, 219822),
            (225037, 225045),
            (225173, 225203),
            (226631, 226828),
            (244240, 244251),
            (247508, 247532),
            (272924, 272942),
            (274365, 274384),
        ],
    }
    linelist = {
        "AU_Mic": "au_mic.lin",
        "HN_Peg": "hn_peg.lin",
        "Eps_Eri": "eps_eri.lin",
        "HD_102195": "hd_102195.lin",
        "HD_130322": "hd_130322.lin",
        "HD_179949": "hd_179949.lin",
        "HD_189733": "hd_189733.lin",
        "55_Cnc": "55_cnc.lin",
        "WASP-18": "wasp_18.lin",
    }

    def parallel(target):
        print(f"Starting {target}")
        segments = range(6, 31)

        # Start the logging to the file
        examples_dir = dirname(realpath(__file__))
        date_string = datetime.datetime.now().isoformat().replace(":", ".")
        log_file = os.path.join(examples_dir, f"logs/{target}_{date_string}.log")
        util.start_logging(log_file)
        # Create the first synthethic spectrum to use for manual masking
        mask = masked.get(target)
        ll = linelist.get(target, "harps.lin")
        sme = run(target, mask=mask, linelist=ll, segments=segments)
        # Add the mask
        mask_file = join(
            examples_dir,
            "results_spline/HD_22049_mask_new_out_monh_teff_logg_vmic_vmac_vsini.sme",
        )
        sme_mask = SME.SME_Structure.load(mask_file)
        sme = sme.import_mask(sme_mask, keep_bpm=True)

        # Finally fit it to the data
        sme.meta["object"] = target
        sme = fit(sme, segments=segments)
        return sme

    if len(sys.argv) == 1:
        for target in tqdm(targets):
            parallel(target)
    else:
        target = sys.argv[1]
        parallel(target)

    pass
