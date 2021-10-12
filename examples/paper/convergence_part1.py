""" Minimum working example of an SME script 
"""

from os.path import dirname, join, realpath
import sys


import numpy as np
from pysme import sme as SME
from pysme import util
from pysme.solve import solve
from pysme.nso import load_solar_spectrum
from pysme.linelist.vald import ValdFile
import logging

logger = logging.getLogger("pysme")
# logger.setLevel(logging.CRITICAL)

if __name__ == "__main__":
    # Define the location of all your files
    # this will put everything into the example dir
    target = "sun"
    examples_dir = dirname(realpath(__file__))
    # in_file = join(examples_dir, "../sun_6440_test.inp")
    # in_file = join(examples_dir, "gr8_HARPS_HD148816.inp")
    vald_file = join(examples_dir, "data/harps.lin")
    out_file = join(examples_dir, f"{target}.sme")
    plot_file = join(examples_dir, f"{target}.html")
    log_file = join(examples_dir, f"{target}.log")

    # Use HARPS spectral range
    wave, flux = load_solar_spectrum()
    wmin, wmax = 3780, 6910
    idx = (wave > wmin) & (wave < wmax)
    wave, flux = wave[idx], flux[idx]
    flux[flux < 0] = 0

    sme = SME.SME_Structure(wave=wave, sob=flux)
    sme.uncs = np.sqrt(flux)
    sme.mask = sme.mask_values["line"]

    sme.linelist = ValdFile(vald_file)

    sme.nmu = 7
    sme.vrad = 0
    sme.cscale = None
    sme.vrad_flag = "each"
    sme.cscale_flag = "linear"
    sme.cscale_type = "match"
    sme.atmo.source = "marcs2014.sav"
    sme.abund = "asplund2009"

    # elems = [
    #     "Al",
    #     "Ba",
    #     "Ca",
    #     "C",
    #     "H",
    #     "K",
    #     "Li",
    #     "Mg",
    #     "Mn",
    #     "Na",
    #     "N",
    #     "O",
    #     "Si",
    # ]
    # for elem in elems:
    #     sme.nlte.set_nlte(elem, f"nlte_{elem}_ama51_pysme.grd")
    # sme.nlte.set_nlte("Fe", "marcs2012_Fe2016.grd")

    # Load the parameters from the command line
    if len(sys.argv) > 1:
        param = sys.argv[1:]
    else:
        param = [5000, 4.0, 0]  # 5000.00, 4.00, 0.00
    sme.teff = float(param[0])
    sme.logg = float(param[1])
    sme.monh = float(param[2])

    # try:
    sme = solve(sme, ["teff", "logg", "monh"])
    print(f"{param[0]}, {param[1]}, {param[2]}, {sme.teff}, {sme.logg}, {sme.monh}")
    # except Exception as ex:
    #     print(f"{param[0]}, {param[1]}, {param[2]}, nan, nan, nan")

