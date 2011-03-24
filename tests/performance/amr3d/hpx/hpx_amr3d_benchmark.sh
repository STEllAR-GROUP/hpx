#! /usr/bin/env bash

# HPX amr3d benchmark script

RUNS=3
REFS="[0,1,2,3]"
CORES="[1,2,5,10,20,30,40]"
GRAINS="[10,9,11,8,12]"

HPX_ROOT="/home/wash/hpx/gcc-4.4.4-release"

TIMESTAMP=`date '+%Y.%m.%d_%H.%M.%S'`

LOG="benchmark.$TIMESTAMP.dat"
TIMING="timing.$TIMESTAMP.dat"

$HPX_ROOT/bin/hpx_optsweep.py -w 1800 -o $LOG -r $RUNS              \
  -a 'REF',"$REFS" -a 'CPU',"$CORES" -a 'GRAIN',"$GRAINS"           \
  /usr/bin/time -f 'CPU REF GRAIN %e %x' --output=$TIMING --append  \
  $HPX_ROOT/bin/had_amr3d_client -r -eREF -pgGRAIN.ini -tCPU

