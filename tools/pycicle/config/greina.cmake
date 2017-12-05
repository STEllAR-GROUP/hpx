#  Copyright (c) 2017 John Biddiscombe
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#######################################################################
# These settings control how jobs are launched and results collected
#######################################################################
# the name used to ssh into the machine
set(PYCICLE_MACHINE "greina.cscs.ch")
# the root location of the build/test tree on the machine
set(PYCICLE_ROOT "/scratch/biddisco/pycicle")
# a flag that says if the machine can send http results to cdash
set(PYCICLE_HTTP "TRUE")
# Launch jobs using slurm rather than directly running them on the machine
set(PYCICLE_SLURM "TRUE")

#######################################################################
# These are settings you can use to define anything useful
#######################################################################
set(GCC_VER       "5.3.0")
set(BOOST_VER     "1.65.1")
set(HWLOC_VER     "1.11.8")
set(JEMALLOC_VER  "5.0.1")
set(OTF2_VER      "2.0")
set(PAPI_VER      "5.5.1")
set(BOOST_SUFFIX  "1_65_1")

set(INSTALL_ROOT     "/users/biddisco/apps")
set(BOOST_ROOT       "${INSTALL_ROOT}/boost/${BOOST_VER}")
set(HWLOC_ROOT       "${INSTALL_ROOT}/hwloc/${HWLOC_VER}")
set(JEMALLOC_ROOT    "${INSTALL_ROOT}/jemalloc/${JEMALLOC_VER}")
set(OTF2_ROOT        "${INSTALL_ROOT}/otf2/${OTF2_VER}")
set(PAPI_ROOT        "${INSTALL_ROOT}/papi/${PAPI_VER}")
set(PAPI_INCLUDE_DIR "${INSTALL_ROOT}/papi/${PAPI_VER}/include")
set(PAPI_LIBRARY     "${INSTALL_ROOT}/papi/${PAPI_VER}/lib/libpfm.so")

set(CFLAGS     "-fPIC")
set(CXXFLAGS   "-fPIC -march native-mtune native-ffast-math-std c++14")
set(LDFLAGS    "-dynamic")
set(LDCXXFLAGS "${LDFLAGS} -std c++14")
set(BUILD_PARALLELISM "8")

set(CTEST_SITE "linux(greina)-gcc-${GCC_VER}-Boost-${BOOST_VER}")
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")

#######################################################################
# The string that is used to drive cmake config step
#######################################################################
string(CONCAT CTEST_BUILD_OPTIONS
    " -DHPX_WITH_CXX14=ON "
    " -DHPX_WITH_NATIVE_TLS=ON "
    " -DCMAKE_CXX_FLAGS=${CXXFLAGS} "
    " -DCMAKE_C_FLAGS=${CFLAGS} "
    " -DCMAKE_EXE_LINKER_FLAGS=${LDCXXFLAGS} "
    " -DHWLOC_ROOT=${HWLOC_ROOT} "
    " -DJEMALLOC_ROOT=${JEMALLOC_ROOT} "
    " -DBOOST_ROOT=${BOOST_ROOT} "
    " -DBoost_ADDITIONAL_VERSIONS=${BOOST_VER} "
#    " -DOTF2_ROOT=${OTF2_ROOT} "
#    " -DPAPI_ROOT=${PAPI_ROOT} "
#    " -DPAPI_INCLUDE_DIR=${PAPI_INCLUDE_DIR} "
#    " -DPAPI_LIBRARY=${PAPI_LIBRARY} "
    " -DHPX_WITH_MALLOC=JEMALLOC "
    " -DHPX_WITH_EXAMPLES=ON "
    " -DHPX_WITH_TESTS=ON "
    " -DHPX_WITH_TESTS_BENCHMARKS=ON "
    " -DHPX_WITH_TESTS_EXTERNAL_BUILD=OFF "
    " -DHPX_WITH_TESTS_HEADERS=OFF "
    " -DHPX_WITH_TESTS_REGRESSIONS=ON "
    " -DHPX_WITH_TESTS_UNIT=ON "
    " -DHPX_WITH_PARCELPORT_MPI=OFF "
    " -DHPX_WITH_THREAD_IDLE_RATES=ON "
    " -DDART_TESTING_TIMEOUT=45 "
)

#######################################################################
# Setup a slurm job submission template
# note that this is intentionally multiline
#######################################################################
set(PYCICLE_SLURM_TEMPLATE "#!/bin/bash
#SBATCH --job-name=hpx-${PYCICLE_PR}
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=long
##SBATCH --distribution=cyclic:cyclic

# module command not in path on greina
source /etc/profile.d/modules.sh

# ---------------------
# unload or load modules that differ from the defaults on the system
# ---------------------
module load slurm
module load git
module load CMake
module unload gcc
module load GCC/${GCC_VER}
module load MVAPICH2/2.2b-GCC-${GCC_VER}

export CC=/cm/shared/apps/easybuild/software/GCC/${GCC_VER}/bin/gcc
export CXX=/cm/shared/apps/easybuild/software/GCC/${GCC_VER}/bin/g++

")
