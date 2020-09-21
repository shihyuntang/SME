from scipy.io import readsav
import numpy as np

from os.path import basename

from .atmosphere import AtmosphereGrid


class SavFile(AtmosphereGrid):
    """ IDL savefile atmosphere grid """

    def __new__(cls, filename):
        data = readsav(filename)

        npoints = data["atmo_grid_maxdep"]
        ngrids = data["atmo_grid_natmo"]
        self = super(SavFile, cls).__new__(cls, ngrids, npoints)

        filename = basename(filename)
        self.source = filename

        # TODO cover all cases
        if "marcs" in filename:
            self.citation_info = r"""
                @ARTICLE{2008A&A...486..951G,
                    author = {{Gustafsson}, B. and {Edvardsson}, B. and {Eriksson}, K. and
                    {J{\o}rgensen}, U.~G. and {Nordlund}, {\r{A}}. and {Plez}, B.},
                    title = "{A grid of MARCS model atmospheres for late-type stars. I. Methods and general properties}",
                    journal = {Astronomy and Astrophysics},
                    keywords = {stars: atmospheres, Sun: abundances, stars: fundamental parameters, stars: general, stars: late-type, stars: supergiants, Astrophysics},
                    year = "2008",
                    month = "Aug",
                    volume = {486},
                    number = {3},
                    pages = {951-970},
                    doi = {10.1051/0004-6361:200809724},
                    archivePrefix = {arXiv},
                    eprint = {0805.0554},
                    primaryClass = {astro-ph},
                    adsurl = {https://ui.adsabs.harvard.edu/abs/2008A&A...486..951G},
                    adsnote = {Provided by the SAO/NASA Astrophysics Data System}}
            """
        elif "atlas" in filename:
            self.citation_info = r"""
                @MISC{2017ascl.soft10017K,
                    author = {{Kurucz}, Robert L.},
                    title = "{ATLAS9: Model atmosphere program with opacity distribution functions}",
                    keywords = {Software},
                    year = "2017",
                    month = "Oct",
                    eid = {ascl:1710.017},
                    pages = {ascl:1710.017},
                    archivePrefix = {ascl},
                    eprint = {1710.017},
                    adsurl = {https://ui.adsabs.harvard.edu/abs/2017ascl.soft10017K},
                    adsnote = {Provided by the SAO/NASA Astrophysics Data System}}
            """
        else:
            self.citation_info = ""  # ???

        atmo_grid = data["atmo_grid"]

        if "RADIUS" in atmo_grid.dtype.names and "HEIGHT" in atmo_grid.dtype.names:
            self.geom = "SPH"
            self["radius"] = atmo_grid["radius"]
            self["height"] = np.stack(atmo_grid["height"])
            # If the radius is given in absolute values
            # self["radius"] /= np.max(self["radius"])
        else:
            self.geom = "PP"
            self["radius"][:] = 1

        self.abund_format = "sme"

        # Scalar Parameters (one per atmosphere)
        self["teff"] = atmo_grid["teff"]
        self["logg"] = atmo_grid["logg"]
        self["monh"] = atmo_grid["monh"]
        self["vturb"] = atmo_grid["vturb"]
        self["lonh"] = atmo_grid["lonh"]
        self["wlstd"] = atmo_grid["wlstd"]
        # Vector Parameters (one array per atmosphere)
        self["rhox"] = np.stack(atmo_grid["rhox"])
        self["tau"] = np.stack(atmo_grid["tau"])
        self["temp"] = np.stack(atmo_grid["temp"])
        self["rho"] = np.stack(atmo_grid["rho"])
        self["xne"] = np.stack(atmo_grid["xne"])
        self["xna"] = np.stack(atmo_grid["xna"])
        self["abund"] = np.stack(atmo_grid["abund"])
        self["opflag"] = np.stack(atmo_grid["opflag"])
        return self
