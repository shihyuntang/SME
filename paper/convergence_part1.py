# -*- coding: utf-8 -*-
""" Minimum working example of an SME script
"""

import logging
import sys
from os.path import dirname, join, realpath

import numpy as np

from pysme import sme as SME
from pysme.gui.plot_plotly import FinalPlot
from pysme.linelist.vald import ValdFile
from pysme.nso import load_solar_spectrum
from pysme.solve import solve

logger = logging.getLogger("pysme")
# logger.setLevel(logging.CRITICAL)

if __name__ == "__main__":
    # Define the location of all your files
    # this will put everything into the example dir
    TARGET = "sun"
    examples_dir = dirname(realpath(__file__))
    # in_file = join(examples_dir, "../sun_6440_test.inp")
    # in_file = join(examples_dir, "gr8_HARPS_HD148816.inp")
    vald_file = join(examples_dir, "data/harps.lin")
    out_file = join(examples_dir, f"{TARGET}.sme")
    plot_file = join(examples_dir, f"{TARGET}.html")
    log_file = join(examples_dir, f"{TARGET}.log")

    # Use HARPS spectral range
    wave, flux = load_solar_spectrum()
    wmin, wmax = 3780, 6910
    idx = (wave > wmin) & (wave <= wmax)
    wave, flux = wave[idx], flux[idx]
    flux[flux < 0] = 0

    # Split into segments
    NPOINTS = 10_000
    nseg = int(np.ceil(wave.size / NPOINTS))
    wave = [wave[i * NPOINTS : (i + 1) * NPOINTS] for i in range(nseg)]
    flux = [flux[i * NPOINTS : (i + 1) * NPOINTS] for i in range(nseg)]

    segments = [10]
    wave = [wave[10]]
    flux = [flux[10]]

    sme = SME.SME_Structure(wave=wave, sob=flux)
    sme.uncs = np.sqrt(sme.spec)
    sme.mask = sme.mask_values["line"]
    sme.nmu = 7

    # Add linelist
    sme.linelist = ValdFile(vald_file)
    sme.linelist.medium = "air"
    wmin = sme.wave[0, 0]
    wmax = sme.wave[0, -1]
    sme.linelist = sme.linelist.trim(wmin, wmax, 100)

    # Add abundances
    sme.abund = "asplund2009"

    # Add atmosphere grid
    sme.atmo.source = "marcs2012.sav"
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

    # Set continuum and radial velocity settings
    sme.vrad = None
    sme.cscale = None
    sme.vrad_flag = "fix"
    sme.cscale_flag = "none"
    sme.cscale_type = "match"
    sme.vrad_limit = 200
    sme.vrad = 0.26897

    # Load the parameters from the command line
    if len(sys.argv) > 1:
        param = sys.argv[1:]
    else:
        param = [5000, 4.4, 0.4]  # 5000.00, 4.00, 0.00
    sme.teff = float(param[0])
    sme.logg = float(param[1])
    sme.monh = float(param[2])
    sme.vmic = 1
    sme.vmac = 3.4
    sme.vsini = 0.5

    # Like HARPS?
    sme.iptype = "gauss"
    sme.ipres = 100_500 * 2

    # try:
    fp = ["teff", "logg", "monh"]
    sme = solve(sme, fp)
    sme.save(
        join(
            examples_dir,
            f"results/convergence_teff_{param[0]}_logg_{param[1]}_monh_{param[2]}.sme",
        )
    )
    print(f"{param[0]}, {param[1]}, {param[2]}, {sme.teff}, {sme.logg}, {sme.monh}")

    plot_file = join(
        examples_dir,
        f"results/convergence_teff_{param[0]}_logg_{param[1]}_monh_{param[2]}.html",
    )
    fig = FinalPlot(sme)
    fig.save(filename=plot_file)
    # except Exception as ex:
    #     print(f"{param[0]}, {param[1]}, {param[2]}, nan, nan, nan")
