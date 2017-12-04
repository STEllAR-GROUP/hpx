#  Copyright (c) 2017 John Biddiscombe
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

#######################################################################
# For debugging this script
#######################################################################
message("Pull request is  " ${PYCICLE_PR})
message("PR-Branchname is " ${PYCICLE_BRANCH})
message("master branch is " ${PYCICLE_MASTER})
message("Machine name is  " ${PYCICLE_HOST})
message("PYCICLE_ROOT is  " ${PYCICLE_ROOT})
message("Random string is " ${PYCICLE_RANDOM})
message("COMPILER is      " ${PYCICLE_COMPILER})
message("BOOST is         " ${PYCICLE_BOOST})

#######################################################################
# Load machine specific settings
#######################################################################
include(${CMAKE_CURRENT_LIST_DIR}/config/${PYCICLE_HOST}.cmake)

#######################################################################
# Generate a slurm job script and launch it
# we must pass all the parms we received through to the slurm script
#######################################################################
string(CONCAT SLURM_TEMPLATE ${SLURM_TEMPLATE}
  "ctest "
  "-S ${PYCICLE_ROOT}/repo/tools/pycicle/dashboard_script.cmake "
  "-DPYCICLE_ROOT=${PYCICLE_ROOT} "
  "-DPYCICLE_HOST=${PYCICLE_HOST} "
  "-DPYCICLE_PR=${PYCICLE_PR} "
  "-DPYCICLE_BRANCH=${PYCICLE_BRANCH} "
  "-DPYCICLE_COMPILER=${PYCICLE_COMPILER} "
  "-DPYCICLE_BOOST=${PYCICLE_BOOST} "
  "-DPYCICLE_MASTER=${PYCICLE_MASTER} \n"
)

file(WRITE "${PYCICLE_ROOT}/build/ctest-slurm-${PYCICLE_RANDOM}.sh" ${SLURM_TEMPLATE})

#######################################################################
# Launch the dashboard test using slurm
# 1 Cancel any build using the same name as this one so that multiple
#   pushes to the same branch are handled cleanly
# 2 Wipe any ctest XML files in the build tree to make it easier to
#   get the most up-to-date one?
# 3 Spawn a new build
#######################################################################
execute_process(
  COMMAND bash "-c" "scancel $(squeue -n hpx-${PYCICLE_PR} -h -o %A) > /dev/null 2>&1;
                     sbatch ${PYCICLE_ROOT}/build/ctest-slurm-${PYCICLE_RANDOM}.sh"
)

file(REMOVE "${PYCICLE_ROOT}/build/ctest-slurm-${PYCICLE_RANDOM}.sh")
