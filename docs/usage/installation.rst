Installation
=============

How to install SME:

Prerequisites:
    - libgfortran5 (e.g., from https://pkgs.org/download/libgfortran5)
    - gcc

0. (optional) create a virtual environment, and activate it.
    - download the environment.yml from https://github.com/AWehrhahn/SME.git
    - cd to the download location
    - `conda env create -f environment.yml`
    - `source activate pysme`

1. Use pip
    - For the "stable" version:
        - pip install pysme-astro
    - For the latest version use:
        - pip install git+https://github.com/AWehrhahn/SME.git

2. (optional) Download data files as part of IDL SME from http://www.stsci.edu/~valenti/sme.html
    - The atmosphere and nlte data files should be downloaded from the server correctly (assuming it is available) if not you can manually copy them into their respective storage locations in ~/.sme/atmospheres and ~/.sme/nlte_grids
        - atmospheres
            - everything from SME/atmospheres
        - nlte_grids
            - \*.grd from SME/NLTE

3. Run SME
    - An simple minimum example is provided in the examples directory (https://github.com/AWehrhahn/SME/tree/master/examples). Make sure to also download the provided input structure.
    - You can then run it with: `python minimum.py`
