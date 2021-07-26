from os.path import join, dirname
import numpy as np
from pysme.sme import SME_Structure
from pysme.nso import load_solar_spectrum
from pysme.abund import Abund
from pysme.linelist.vald import ValdFile
from pysme.synthesize import synthesize_spectrum

if __name__ == "__main__":
    wave, spec = load_solar_spectrum(ravel=False)
    spec = np.clip(spec, 0, 1)

    # Only use from 6022 Å to 7712 Å
    select = slice(500, 700)
    wave = wave[select]
    spec = spec[select]

    sme = SME_Structure()
    sme.wave = wave
    sme.spec = spec
    sme.uncs = np.sqrt(spec)

    sme.teff = 5770
    sme.logg = 4.4
    sme.abund = Abund.solar()
    sme.vmic = 1
    sme.vmac = 2
    sme.vsini = 1

    vald_file = join(dirname(__file__), "data/sun_full.lin")
    sme.linelist = ValdFile(vald_file)

    sme.cscale_flag = "none"
    sme.vrad_flag = "whole"

    sme = synthesize_spectrum(sme)

    save_file = join(dirname(__file__), "results/sun.sme")
    sme.save(save_file)

    pass
