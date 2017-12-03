#!/bin/bash

ssh $1 "ctest -S $2/repo/tools/pycicle/dashboard_slurm.cmake " \
"-DPYCICLE_ROOT=$2 -DPYCICLE_HOST=$3 -DPYCICLE_PR=$4 " \
"-DPYCICLE_RANDOM=$5 -DPYCICLE_COMPILER=$6 -DPYCICLE_BOOST=$7" \
"-DPYCICLE_MASTER=master "

#ssh $1 "echo ctest -S $2/repo/tools/pycicle/dashboard_script.cmake " \
#"-DPYCICLE_ROOT=$2 -DPYCICLE_HOST=$3 -DPYCICLE_PR=$4 " \
#"-DPYCICLE_RANDOM=$5 -DPYCICLE_COMPILER=$6 -DPYCICLE_BOOST=$7" \
#"-DPYCICLE_MASTER=master "
