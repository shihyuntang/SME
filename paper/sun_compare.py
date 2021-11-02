# -*- coding: utf-8 -*-
""" Minimum working example of an SME script
"""
import os.path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import readsav

from pysme import sme as SME
from pysme import util
from pysme.abund import Abund
from pysme.gui import plot_plotly
from pysme.linelist.vald import ValdFile
from pysme.persistence import save_as_idl
from pysme.solve import solve
from pysme.synthesize import synthesize_spectrum

if __name__ == "__main__":

    # Define the location of all your files
    # this will put everything into the example dir
    target = "sun"
    examples_dir = os.path.dirname(os.path.realpath(__file__))
    in_file = os.path.join(examples_dir, "data/sun_6440_test.inp")
    idl_file = os.path.join(examples_dir, "data/sun_6440_test.out")
    out_file = os.path.join(examples_dir, f"results/{target}.sme")
    plot_file = os.path.join(examples_dir, f"results/{target}.html")
    log_file = os.path.join(examples_dir, f"results/{target}.log")

    # Start the logging to the file
    util.start_logging(log_file)

    # Load your existing SME structure or create your own
    sme = SME.SME_Structure.load(in_file)
    idl = SME.SME_Structure.load(idl_file)

    sme.cscale = [0, 1]
    sme.cscale_type = "match+mask"
    sme.vrad = idl.vrad
    sme.vrad_flag = "none"
    sme.cscale_flag = "none"

    sme = synthesize_spectrum(sme)

    # Save results
    sme.save(out_file)

    # Plot results
    fig = plot_plotly.FinalPlot(
        sme, orig=idl.synth, labels={"synth": "PySME", "orig": "IDL SME"}
    )
    fig.save(filename=plot_file)
    pass
