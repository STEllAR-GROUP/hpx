#!/bin/bash

ssh $1 "ctest -S $2/repo/tools/pycicle/dashboard_slurm.cmake " \
"-DPYCICLE_ROOT=$2 -DPYCICLE_HOST=$3 " \
"-DPYCICLE_PR=$4 -DPYCICLE_BRANCH=$5 " \
"-DPYCICLE_RANDOM=$6 -DPYCICLE_COMPILER=$7 -DPYCICLE_BOOST=$8" \
"-DPYCICLE_MASTER=master "
