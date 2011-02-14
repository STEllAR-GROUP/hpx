#! /usr/bin/env bash
#
# Copyright (c) 2010-2011 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# TODO: convert to python script, Bash is terribly non-portable

export HPX_RUN=$HOME/projects/parallex_svn/projects/hpx_run

cd $HOME
echo Job $PBS_JOBNAME is executing in `pwd` with $PBS_NODEFILE as the node file:
echo $PATH
echo $LD_LIBRARY_PATH
date
for T in 1 2 4 8; do 
  for L in 1 2 4 8 16 32; do
    echo "Running with -l $L:$T"
    python $HPX_RUN/hpx_run.py -m $PBS_NODEFILE -l $L:$T "cpi_test --throws=1000000 --granularity=$T" -s;
  done;
done
date

