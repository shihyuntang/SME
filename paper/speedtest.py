# -*- coding: utf-8 -*-
""" Minimum working example of an SME script
"""

import gc
import os.path
import time
from os.path import dirname, join, realpath

import numpy as np

from pysme import sme as SME
from pysme import util
from pysme.abund import Abund
from pysme.linelist.vald import ValdFile
from pysme.persistence import save_as_idl
from pysme.synthesize import synthesize_spectrum

if __name__ == "__main__":
    # Define the location of all your files
    # this will put everything into the example dir
    target = "sun"
    examples_dir = join(dirname(realpath(__file__)), "..")
    in_file = join(examples_dir, "sun_6440_test.inp")
    out_file = join(examples_dir, f"{target}.sme")
    plot_file = join(examples_dir, f"{target}.html")
    log_file = join(examples_dir, f"{target}.log")

    # Start the logging to the file
    util.start_logging(log_file)

    # Load your existing SME structure or create your own
    sme = SME.SME_Structure.load(in_file)

    sme.abund = Abund(0, "asplund2009")
    sme.linelist = ValdFile(join(examples_dir, "sun.lin"))
    sme.nmu = 7
    sme.vrad = 0
    sme.cscale = None
    sme.vrad_flag = "none"
    sme.cscale_flag = "none"
    sme.cscale_type = "match"

    sme.atmo.source = "marcs2012.sav"

    # save_as_idl(sme, "speedtest.inp")
    # Run it once to load the atmosphere
    start = time.time()
    sme = synthesize_spectrum(sme)
    end = time.time()
    runtime = end - start
    print(f"Single Runtime: {runtime} s")

    gc.disable()
    runtime = []
    for i in range(1000):
        start = time.time()
        synthesize_spectrum(sme)
        end = time.time()
        runtime += [end - start]

    print(
        f"Runtime: {np.mean(runtime)} s +- {np.std(runtime)}, min {np.min(runtime)} s"
    )
