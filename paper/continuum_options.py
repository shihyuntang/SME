# -*- coding: utf-8 -*-
""" Minimum working example of an SME script
"""

from os.path import dirname, join, realpath

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import readsav
from scipy.ndimage import label as scipy_label

from pysme import sme as SME
from pysme import util
from pysme.abund import Abund
from pysme.linelist.vald import ValdFile
from pysme.persistence import save_as_idl
from pysme.solve import solve
from pysme.synthesize import synthesize_spectrum

if __name__ == "__main__":
    # Define the location of all your files
    # this will put everything into the example dir
    examples_dir = dirname(realpath(__file__))
    mask_file = join(examples_dir, "continuum.sme")
    in_file = join(examples_dir, "gr8_HARPS_HD148816.inp")

    # Load your existing SME structure or create your own
    sme = SME.SME_Structure.load(in_file)
    sme_mask = SME.SME_Structure.load(mask_file)

    sme.mask = sme_mask.mask
    # sme.nmu = 7
    # sme.teff = 5770
    # sme.logg = 4.4
    # sme.abund = Abund(0, "asplund2009")
    # sme.vmic = 1
    # sme.vmac = 2
    # sme.vsini = 2

    # sme.atmo.source = "marcs2014.sav"
    # sme.linelist = ValdFile(join(examples_dir, "sun.lin"))

    # orig = np.copy(sme.synth[0])

    # Start SME solver
    sme.cscale = None
    sme.vrad_flag = "each"

    continuum = {}
    synth = {}
    x = sme.wave[0] - sme.wave[0][0]
    # Mask linear
    sme.cscale_type = "mask"
    sme.cscale_flag = "linear"
    sme.cscale = None
    sme.vrad = None
    sme = synthesize_spectrum(sme, segments=[0])
    continuum["mask+linear"] = np.polyval(sme.cscale[0], x)
    synth["mask+linear"] = np.copy(sme.synth[0])
    # Mask quadratic
    sme.cscale_type = "mask"
    sme.cscale_flag = "quadratic"
    sme.cscale = None
    sme.vrad = None
    sme = synthesize_spectrum(sme, segments=[0])
    continuum["mask+quadratic"] = np.polyval(sme.cscale[0], x)
    synth["mask+quadratic"] = np.copy(sme.synth[0])
    # Match linear
    sme.cscale_type = "match"
    sme.cscale_flag = "linear"
    sme.cscale = None
    sme.vrad = None
    sme = synthesize_spectrum(sme, segments=[0])
    continuum["match+linear"] = np.polyval(sme.cscale[0], x)
    synth["match+linear"] = np.copy(sme.synth[0])
    # Match quadratic
    sme.cscale_type = "match"
    sme.cscale_flag = "quadratic"
    sme.cscale = None
    sme.vrad = None
    sme = synthesize_spectrum(sme, segments=[0])
    continuum["match+quadratic"] = np.polyval(sme.cscale[0], x)
    synth["match+quadratic"] = np.copy(sme.synth[0])
    # Match+Mask linear
    sme.cscale_type = "match+mask"
    sme.cscale_flag = "linear"
    sme.cscale = None
    sme.vrad = None
    sme = synthesize_spectrum(sme, segments=[0])
    continuum["match+mask+linear"] = np.polyval(sme.cscale[0], x)
    synth["match+mask+linear"] = np.copy(sme.synth[0])
    # Match+Mask quadratic
    sme.cscale_type = "match+mask"
    sme.cscale_flag = "quadratic"
    sme.cscale = None
    sme.vrad = None
    sme = synthesize_spectrum(sme, segments=[0])
    continuum["match+mask+quadratic"] = np.polyval(sme.cscale[0], x)
    synth["match+mask+quadratic"] = np.copy(sme.synth[0])
    # Spline
    sme.cscale_type = "spline"
    sme.cscale_flag = 2
    sme.cscale = None
    sme.vrad = None
    sme = synthesize_spectrum(sme, segments=[0])
    continuum["spline"] = sme.cscale[0]
    synth["spline"] = np.copy(sme.synth[0])
    # Spline+Mask
    sme.cscale_type = "spline+mask"
    sme.cscale_flag = 2
    sme.cscale = None
    sme.vrad = None
    sme = synthesize_spectrum(sme, segments=[0])
    continuum["spline+mask"] = sme.cscale[0]
    synth["spline+mask"] = np.copy(sme.synth[0])
    # MCMC
    # sme.cscale_type = "mcmc"
    # sme.cscale_flag = "linear"
    # sme.cscale = None
    # sme.vrad = None
    # sme = synthesize_spectrum(sme, segments=[0])
    # continuum["mcmc+linear"] = np.polyval(sme.cscale[0], x)

    # Add last calculate the spectrum without continuum correction
    sme.cscale_type = "mask"
    sme.cscale_flag = "none"
    sme = synthesize_spectrum(sme, segments=[0])

    # Plot results
    for label, cont in continuum.items():

        plot_file = join(dirname(__file__), f"images/continuum_{label}.png")
        plt.plot(sme.wave[0], sme.spec[0], label="Observation")
        # plt.plot(sme.wave[0], sme.synth[0], label="Synthetic")

        m = sme.mask[0] == 2
        labels, n = scipy_label(m)
        for i in range(1, n):
            mask = labels == i
            plt.plot(
                sme.wave[0][mask],
                sme.spec[0][mask],
                color="tab:red",
                label="Mask" if i == 1 else None,
            )

        plt.plot(sme.wave[0], cont, label=f"{label} Continuum")
        plt.plot(
            sme.wave[0],
            synth[label],
            label=f"{label} Corrected",
            color="tab:purple",
        )

        plt.legend(loc="lower left", fontsize="small")
        plt.xlabel("Wavelength [Å]")
        plt.ylabel("Flux [A.U.]")
        # plt.ylim(0.9, 1.01)
        plt.savefig(plot_file)
        plt.clf()

    # plot_file = join(dirname(__file__), "images/continuum_2.png")
    # plt.plot(sme.wave[0], sme.spec[0], label="Observation")
    # plt.plot(sme.wave[0], sme.synth[0], label="Synthetic")
    # plt.fill_between(
    #     sme.wave[0],
    #     0,
    #     sme.spec[0],
    #     where=sme.mask[0] == 1,
    #     label="Mask Line",
    #     facecolor="#bcbd22",
    #     alpha=1,
    # )

    # m = sme.mask[0] == 2
    # m[1:] = m[:-1] | m[1:]
    # m[:-1] = m[:-1] | m[1:]
    # plt.fill_between(
    #     sme.wave[0],
    #     0,
    #     sme.spec[0],
    #     where=m,
    #     label="Mask Continuum",
    #     facecolor="#d62728",
    #     alpha=1,
    # )

    # for label, cont in continuum.items():
    #     plt.plot(sme.wave[0], sme.synth[0] * cont, label=label)
    # plt.legend(loc="lower right", fontsize="small")
    # plt.xlabel("Wavelength [Å]")
    # plt.ylabel("Flux [A.U.]")
    # plt.ylim(0.9925, 1.004)
    # plt.savefig(plot_file)
    # plt.show()

    pass
