# -*- coding: utf-8 -*-
from os.path import dirname, join
from re import I

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
    # wave, spec = load_solar_spectrum(ravel=False)
    # spec = np.clip(spec, 0, 1)

    # # Only use from 6022 Å to 7712 Å
    # select = slice(500, 700)
    # wave = wave[select]
    # spec = spec[select]

    # log_file = join(dirname(__file__), "logs/sun.log")
    # start_logging(log_file)

    # sme = SME_Structure()
    # sme.wave = wave
    # sme.spec = spec
    # sme.uncs = np.sqrt(spec)

    # sme.teff = 5770
    # sme.logg = 4.4
    # sme.abund = Abund.solar()
    # sme.vmic = 1
    # sme.vmac = 2
    # sme.vsini = 1

    # vald_file = join(dirname(__file__), "data/sun_full.lin")
    # sme.linelist = ValdFile(vald_file)

    # sme.cscale_flag = "none"
    # sme.vrad_flag = "whole"

    # sme.atmo.source = "marcs2012.sav"
    # sme.atmo.depth = "RHOX"
    # sme.atmo.interp = "TAU"
    # sme.mask = 1
    # idl_file = join(dirname(__file__), "sun.inp")
    # save_as_idl(sme, idl_file)
    # # sme = synthesize_spectrum(sme)
    # sme = solve(sme, ["teff", "logg", "monh"])
    # sme.save(save_file)

    save_file = join(dirname(__file__), "results/sun.sme")
    sme = SME_Structure.load(save_file)

    idl_file = join(dirname(__file__), "sun3.out")
    idl = SME_Structure.load(idl_file)

    plot_file = join(dirname(__file__), "results/sun.html")
    fig = plot_plotly.FinalPlot(sme, orig=idl.synth)
    fig.save(filename=plot_file, auto_open=False)

    pass
