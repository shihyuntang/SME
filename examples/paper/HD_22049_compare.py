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
from scipy.io import readsav
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
    target = "HD_22049"
    sdb = StellarDB()
    # star = sdb.auto_fill(target)
    star = sdb.load(target)
    alias = [re.sub(r"[-_ ]", "", s).lower() for s in star["id"]]

    examples_dir = dirname(realpath(__file__))
    data_dir = join(examples_dir, "data")

    in_file = os.path.join(examples_dir, f"results/epseri2.out")
    out_file = os.path.join(examples_dir, f"results/{target}_compare_out.sme")
    plot_file = os.path.join(examples_dir, f"results/{target}_compare_out.html")
    date_string = datetime.datetime.now().isoformat().replace(":", ".")
    log_file = os.path.join(examples_dir, f"results/{target}_{date_string}.log")

    # Start the logging to the file
    util.start_logging(log_file)

    # Run SME
    sme = SME.SME_Structure.load(in_file)
    orig = sme.synth.copy()
    sme.cscale_flag = "linear"
    sme.cscale_type = "match+mask"
    sme.cscale[0] = [0, 1]

    rvel = 100
    wmin, wmax = sme.wran[0]
    wmin *= 1 - rvel / 3e5
    wmax *= 1 + rvel / 3e5
    sme.linelist = sme.linelist.trim(wmin, wmax)

    sme.specific_intensities_only = True
    wmod, smod, cmod = synthesize_spectrum(sme, segments=[0])

    orig = readsav(in_file)["sme"]
    sint = orig["SINT"][0]
    wint = orig["JINT"][0]
    wind = orig["WIND"][0]

    sint = sint[0, 0 : wind[0]]
    wint = wint[0 : wind[0]]

    plt.plot(wint, sint)
    plt.plot(wmod[0], smod[0] * cmod[0])
    plt.show()

    # Plot results
    fig = plot_plotly.FinalPlot(sme, orig=orig)
    fig.save(filename=plot_file)
    print(f"Finished: {target}")
