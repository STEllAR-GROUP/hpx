#! /usr/bin/env bash

################################################################################
# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
################################################################################

###############################################################################
# configuration 

RUNS=3
REFS=(0 1 2 3 4)
CORES=(1 2 4 5 6 8 10 12 16 18 20 24 30 32 36 40 42)

MPI_ROOT="/opt/mpich2/1.2.1p1/gcc-4.4.4"
CACTUS_ROOT="/home/wash/vcs/cactus-gcc/Cactus"

###############################################################################
TIMESTAMP=`date '+%Y.%m.%d_%H.%M.%S'`

LOG="benchmark.$TIMESTAMP.dat"
OUT="timing.$TIMESTAMP.dat"

[ $INTEL_ROOT ] && source $INTEL_ROOT/bin/ifortvars.sh intel64

for REF in ${REFS[@]}
do
  for CPU in ${CORES[@]}
  do
    TIME_CMD="/usr/bin/time --output=$OUT --append"
    MPIEXEC_CMD="$MPI_ROOT/bin/mpiexec -np $CPU"
    INPUT="wavetoy_lor$REF.par"
    for ((I = 0; I <= ${RUNS}; I += 1))
    do
      echo "=================================================" >> $LOG 
      echo -n "$TIME_CMD -f \"$CPU $REF N %e %x\" $MPIEXEC_CMD" >> $LOG
      echo "$CACTUS_ROOT/exe/cactus_sim3 $INPUT" >> $LOG 
      $TIME_CMD -f "$CPU $REF N %e %x" $MPIEXEC_CMD      \
        $CACTUS_ROOT/exe/cactus_sim3 $INPUT >> $LOG 2>&1
    done
  done
done

