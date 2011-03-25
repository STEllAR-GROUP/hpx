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
REFS="[0,1,2,3]"
CORES="[1,2,4,6,8,12,16,18,24,30,32,36,40,42]"
GRAINS="[10,9,11,8,12]"

HPX_ROOT="/home/wash/hpx/gcc-4.4.4-release"

###############################################################################

TIMESTAMP=`date '+%Y.%m.%d_%H.%M.%S'`

LOG="benchmark.$TIMESTAMP.dat"
OUT="timing.$TIMESTAMP.dat"

$HPX_ROOT/bin/hpx_optsweep.py -w 1800 -o $LOG -r $RUNS             \
  -a 'REF',"$REFS" -a 'CPU',"$CORES" -a 'GRAIN',"$GRAINS"          \
  /usr/bin/time -f 'CPU REF GRAIN %%e %%x' --output=$OUT --append  \
  $HPX_ROOT/bin/had_amr3d_client -r -eREF -pgGRAIN.ini -tCPU

