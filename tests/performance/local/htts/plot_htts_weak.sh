#! /bin/bash

gnuplot -e "INPUT=\"$1.dat\"" -e "OUTPUT=\"$1.pdf\"" -e "HEADER=\"$1.gpi\"" $PWD/plot_htts_weak.gpi 


