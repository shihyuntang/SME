# -*- coding: utf-8 -*-
import json
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from pysme.sme import SME_Structure

if __name__ == "__main__":
    results_dir = join(dirname(__file__), "json")
    targets = [
        # "au_mic.json",
        "eps_eri.json",
        "55_cnc.json",
        "hd_102195.json",
        "hd_130322.json",
        "hd_179949.json",
        "hd_189733.json",
        "hn_peg.json",
        "wasp-18.json",
    ]
    names = np.array(
        [
            # "AU Mic",
            "Eps Eri",
            "55 Cnc",
            "HD 102195",
            "HD 130322",
            "HD 179949",
            "HD 189733",
            "HN Peg",
            "WASP-18",
        ]
    )
    snr = []
    parameters = ["teff", "logg", "monh", "vmic", "vmac", "vsini"]
    method = []
    units = ["K", "log(cm/s**2)", "[F/H]", "km/s", "km/s", "km/s"]

    values = {p: np.zeros(len(targets)) for p in parameters}
    uncertainties = {p: np.zeros(len(targets)) for p in parameters}
    inf_values = {p: np.zeros(len(targets)) for p in parameters}
    inf_uncertainties = {p: np.zeros(len(targets)) for p in parameters}

    # Load the data from targets
    for i, target in enumerate(targets):
        fname = join(results_dir, target)
        with open(fname) as f:
            sme = json.load(f)
        # sme = SME_Structure.load(fname)
        for p in parameters:
            values[p][i] = sme[p]
            uncertainties[p][i] = sme[f"unc_{p}"]
            inf_values[p][i] = sme.get(f"inf_{p}", np.nan)
            inf_uncertainties[p][i] = sme.get("inf_unc_{p}", np.nan)
        method += [sme["method"]]
        # fitparameters = np.asarray(sme.fitresults.parameters)
        # fituncertainties = np.asarray(sme.fitresults.uncertainties)
        # for p in parameters:
        #     uncertainties[p][i] = fituncertainties[fitparameters == p]
    method = np.array(method)

    # Plot trends
    for p, u in zip(parameters, units):
        # x = values[p]
        x = values["teff"]
        y = uncertainties[p]

        xf = np.linspace(x.min(), x.max(), 100)
        popt, pcov = curve_fit(
            lambda x, *p: np.polyval(p, x),
            x[names != "AU Mic"],
            y[names != "AU Mic"],
            p0=np.zeros(2),
            method="trf",
            loss="soft_l1",
        )
        err = np.sqrt(np.diag(pcov))
        f = np.polyval(popt, xf)
        # fup = np.polyval(popt+err, xf)
        # flo = np.polyval(popt-err, xf)
        # Plot 1
        plt.plot(x, y, "o")
        plt.plot(xf, f, "--")
        # plt.plot(xf, fup, "r-.")
        # plt.plot(xf, flo, "r-.")

        for i in range(len(targets)):
            plt.text(x[i], y[i], names[i])
        plt.xlabel("$T_{eff}$ [K]")
        plt.ylabel(fr"$\sigma${p} [{u}]")

        plt.savefig(join(dirname(__file__), f"images/trend_{p}.png"))
        # plt.show()
        plt.clf()

        # Plot 2
        if not np.all(np.isnan(inf_values[p])):
            delta = values[p] - inf_values[p]
            xerr = inf_uncertainties[p]
            mask = method == "interferometry"

            plt.errorbar(
                x[mask],
                delta[mask],
                yerr=y[mask],
                xerr=xerr[mask],
                fmt="o",
                label="interferometry",
            )
            plt.errorbar(
                x[~mask],
                delta[~mask],
                yerr=y[~mask],
                xerr=xerr[~mask],
                fmt="o",
                mfc="#ffffff",
                label="spectroscopy",
            )

            plt.hlines(0, x.min(), x.max(), colors="k")
            for i in range(len(targets)):
                if not np.isnan(delta[i]):
                    plt.text(x[i], delta[i], names[i])

            plt.legend(loc="upper left")
            plt.ylabel(fr"$\Delta${p} [{u}]")
            plt.xlabel("$T_{eff}$ [K]")
            plt.savefig(join(dirname(__file__), f"images/delta_{p}.png"))
            # plt.show()
            plt.clf()
        else:
            y = values[p]

            popt, pcov = curve_fit(
                lambda x, *p: np.polyval(p, x),
                x[names != "AU Mic"],
                y[names != "AU Mic"],
                p0=np.zeros(2),
                method="trf",
                loss="soft_l1",
            )
            err = np.sqrt(np.diag(pcov))
            f = np.polyval(popt, xf)
            # fup = np.polyval(popt + err, xf)
            # flo = np.polyval(popt - err, xf)

            plt.plot(x, y, "o")
            plt.plot(xf, f, "--")
            # plt.plot(xf, fup, "r-.")
            # plt.plot(xf, flo, "r-.")

            for i in range(len(targets)):
                plt.text(x[i], y[i], names[i])
            plt.xlabel("$T_{eff}$ [K]")
            plt.ylabel(f"{p} [{u}]")
            plt.savefig(join(dirname(__file__), f"images/teff_trend_{p}.png"))
            # plt.show()
            plt.clf()

    pass
