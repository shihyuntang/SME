#!/bin/bash
parallel --bar --joblog convergence.log --header : --results ./log --linebuffer "python convergence_part1.py {1} {2} {3}" :::: convergence_teffs.txt :::: convergence_loggs.txt :::: convergence_monhs.txt > convergence_data.txt

# parallel --bar --joblog convergence.log --header : --results ./log --linebuffer "python convergence_part1.py {1} {2} {3}" ::: teff 5000 ::: logg 4.40 ::: monh 0.4
