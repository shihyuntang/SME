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

if __name__ == "__main__":
    # Define the location of all your files
    # this will put everything into the example dir
    target = "L_98-59"
    sdb = StellarDB()
    # sdb.auto_fill(target)
    star = sdb.load(target)
    alias = [re.sub(r"[-_ ]", "", s).lower() for s in star["id"]]

    examples_dir = dirname(realpath(__file__))
    data_dir = join(examples_dir, "data")

    # Find the correct data file for this target
    # fname = "ADP.2019-01-30T01:13:58.172.fits"
    fname = "L_98-59_mask_out.sme"
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

    # Set radial velocity and continuum settings
    # Set RV and Continuum flags
    sme.vrad_flag = "fix"
    sme.cscale_flag = "none"
    sme.cscale_type = "match"
    sme.vrad = -14

    # Reset observations
    sme.wave = None
    sme.spec = None
    sme.synth = None
    sme.telluric = None
    # Set new Wavelength ranges of CRIRES+
    # This is for setting L3426
    sme.wran = [
        [2885, 2935],
        [3037, 3090],
        [3206, 3261],
        [3394, 3453],
        [3607, 3669],
        [3847, 3914],
        [4122, 4193],
    ]

    # Perfrom the calculation
    sme = synthesize_spectrum(sme)
    print(sme.citation())

    # Save results
    sme.save(out_file)

    # Plot results
    fig = plot_plotly.FinalPlot(sme)
    fig.save(filename=plot_file)
    print(f"Finished: {target}")
