#!/bin/bash
# Run GNU parallel using the parallel.py script in this folder
# by passing 3 parameters to it (teff, logg, monh)
# For each parameter there are three values, resulting in
# a total of 27 different runs

# Format for the call is:
# parallel cmd ::: p1 ::: p2
# where cmd is the command to be executed (here calling the python script with
# three command line parameters), and p1 and p2 are the parameters to pass to the script
# With keywords:
# --bar : show a progress bar
# --joblog parallel.log : save runtime data about the runs in parallel.log
# --header : : the first value of each parameter list is its header, which is not used otherwise
# --results ./log : save the stdout and stderr of each run to logfiles in the log folder.
#                   These logfiles are seperate from the regular logfiles created by PySME,
#
# The "normal" stdout is passed to dev/null, since we don't need it and only want the progress bar
#
# For the full documentation on GNU parallel see https://www.gnu.org/software/parallel/parallel.html

parallel --bar --joblog parallel.log --header : --results log/ "python parallel.py {teff} {logg} {monh}" ::: teff 5000 5500 6000 ::: logg 4 4.2 4.4 ::: monh -0.5 0 0.5 > /dev/null
