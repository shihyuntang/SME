# -*- coding: utf-8 -*-
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np

from pysme.abund import Abund
from pysme.gui import plot_plotly
from pysme.linelist.vald import ValdFile
from pysme.nso import load_solar_spectrum
from pysme.persistence import save_as_idl
from pysme.sme import SME_Structure
from pysme.solve import solve
from pysme.synthesize import synthesize_spectrum
from pysme.util import start_logging

if __name__ == "__main__":
    wave, spec = load_solar_spectrum(ravel=False)
    spec = np.clip(spec, 0, 1)

    # Only use from 6022 Å to 7712 Å
    select = slice(500, 700)
    wave = wave[select]
    spec = spec[select]

    log_file = join(dirname(__file__), "logs/sun.log")
    save_file = join(dirname(__file__), "results/sun_compare.sme")
    start_logging(log_file)

    sme = SME_Structure()
    sme.wave = wave
    sme.spec = spec
    sme.uncs = np.sqrt(spec)

    sme.teff = 5500
    sme.logg = 5.0
    sme.abund = Abund(0.2, "asplund2009")
    sme.vmic = 1
    sme.vmac = 3
    sme.vsini = 1.6

    vald_file = join(dirname(__file__), "data/sun_full.lin")
    sme.linelist = ValdFile(vald_file)
    # TODO: is this correct?
    sme.linelist.medium = "air"

    sme.cscale_flag = "none"
    sme.vrad_flag = "whole"

    sme.atmo.source = "marcs2012.sav"
    sme.atmo.depth = "RHOX"
    sme.atmo.interp = "TAU"
    sme.mask = 1
    idl_file = join(dirname(__file__), "sun.inp")
    save_as_idl(sme, idl_file)
    # sme = synthesize_spectrum(sme)
    sme = solve(sme, ["teff", "logg", "monh"])
    sme.save(save_file)

    save_file = join(dirname(__file__), "results/sun_compare.sme")
    sme = SME_Structure.load(save_file)

    idl_file = join(dirname(__file__), "sun1.out")
    idl = SME_Structure.load(idl_file)

    # for i in range(30):
    #     plt.hist(
    #         sme.fitresults.residuals[i * 1024 : (i + 1) * 1024]
    #         / sme.fitresults.derivative[i * 1024 : (i + 1) * 1024, 0],
    #         bins="auto",
    #         histtype="step",
    #     )
    # plt.show()

    plot_file = join(dirname(__file__), "results/sun_compare.html")
    fig = plot_plotly.FinalPlot(
        sme, orig=idl.synth, labels={"orig": "IDL SME", "synth": "PySME"}
    )
    fig.save(filename=plot_file, auto_open=False)

    pass
