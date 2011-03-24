#! /usr/bin/env bash

# Carpet amr3d benchmark script

RUNS=3
REFS=(0 1 2 3)
CORES=(1 2 5 10 20 30 40)

INTEL_ROOT="/opt/intel/fc/11.1"
MPI_ROOT="/opt/mpich2/1.2.1p1/intel-11.1/bin/mpiexec"
CACTUS_ROOT="/home/wash/vcs/cactus-intel/Cactus"

TIMESTAMP=`date '+%Y.%m.%d_%H.%M.%S'`

LOG="benchmark.$TIMESTAMP.dat"
TIMING="timing.$TIMESTAMP.dat"

[ $INTEL_ROOT ] && source $INTEL_ROOT/bin/ifortvars.sh intel64

for ((I = 0; I <= ${RUNS}; I += 1))
do
  for REF in ${REFS[@]}
  do
    for CPU in ${CORES[@]}
    do
      TIME_CMD="/usr/bin/time -f \"$CPU $REF - %e %x\" --append -o $TIMING"
      MPIEXEC_CMD="$MPI_ROOT/bin/mpiexec -np $CPU"
      INPUT="wavetoy_lor$REF.par"
      echo "=================================================" >> $LOG 
      echo "$TIME_CMD $MPIEXEC_CMD $CACTUS_ROOT/exe/cactus_sim3 $INPUT" >> $LOG 
      $TIME_CMD $MPIEXEC_CMD $CACTUS_ROOT/exe/cactus_sim3 $INPUT >> $LOG 2>&1 
    done
  done
done

