#  Copyright (c) 2017 John Biddiscombe
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#######################################################################
# These settings control how jobs are launched and results collected
#######################################################################
# the name used to ssh into the machine
set(PYCICLE_MACHINE "daint.cscs.ch")
# the root location of the build/test tree on the machine
set(PYCICLE_ROOT "/scratch/snx3000/biddisco/pycicle")
# a flag that says if the machine can send http results to cdash
set(PYCICLE_HTTP TRUE)
# Launch jobs using slurm rather than directly running them on the machine
set(PYCICLE_SLURM TRUE)

# These versions are ok for gcc or clang
set(BOOST_VER            "1.65.0")
set(HWLOC_VER            "1.11.7")
set(JEMALLOC_VER         "5.0.1")
set(OTF2_VER             "2.0")
set(PAPI_VER             "5.5.1")
set(BOOST_SUFFIX         "1_65_0")
set(CMAKE_VER            "3.9.1")

if (PYCICLE_COMPILER MATCHES "gcc")
  set(GCC_VER             "6.2.0")
  set(PYCICLE_BUILD_STAMP "gcc-${GCC_VER}")
  #
  set(INSTALL_ROOT     "/apps/daint/UES/6.0.UP04/HPX")
  set(BOOST_ROOT       "${INSTALL_ROOT}/boost/${GCC_VER}/${BOOST_VER}")
  #
  set(CFLAGS           "-fPIC")
  set(CXXFLAGS         "-fPIC -march=native -mtune=native -ffast-math -std=c++14")
  set(LDFLAGS          "-dynamic")
  set(LDCXXFLAGS       "${LDFLAGS} -std=c++14")

  # multiline string
  set(PYCICLE_COMPILER_SETUP "
    #
    module load gcc/${GCC_VER}
    #
    # use Cray compiler wrappers to make MPI use easy
    export  CC=/opt/cray/pe/craype/default/bin/cc
    export CXX=/opt/cray/pe/craype/default/bin/CC
    #
    export CFLAGS=\"${CFLAGS}\"
    export CXXFLAGS=\"${CXXFLAGS}\"
    export LDFLAGS=\"${LDFLAGS}\"
    export LDCXXFLAGS=\"${LDCXXFLAGS}\"
  ")

  string(CONCAT CTEST_BUILD_OPTIONS
    "  -DHPX_WITH_CXX14=ON "
  )

elseif(PYCICLE_COMPILER MATCHES "clang")
  set(CLANG_ROOT         "/users/biddisco/apps/daint/llvm")
  set(CMAKE_C_COMPILER   "${CLANG_ROOT}/bin/clang")
  set(CMAKE_CXX_COMPILER "${CLANG_ROOT}/bin/clang++")
  #
  set(PYCICLE_BUILD_STAMP "clang-6.0.0")
  #
  set(OTF2_VER         "2.1")
  #
  set(INSTALL_ROOT     "/users/biddisco/apps/daint/clang")
  set(BOOST_ROOT       "${INSTALL_ROOT}/boost/${BOOST_VER}")
  #
  set(CFLAGS           "-fPIC")
  set(CXXFLAGS         "-fPIC -march=native -mtune=native -ffast-math -std=c++17 -stdlib=libc++ -I${CLANG_ROOT}/include/c++/v1")
  set(LDFLAGS          "-L${CLANG_ROOT}/lib -rpath ${CLANG_ROOT}/lib")
  set(LDCXXFLAGS       "${LDFLAGS} -std=c++17 -stdlib=libc++")

  # multiline string
  set(PYCICLE_COMPILER_SETUP "
    #
    export PATH=${CLANG_ROOT}/bin:$PATH
    export LD_LIBRARY_PATH=${CLANG_ROOT}/lib:$LD_LIBRARY_PATH
    export PATH=${CLANG_ROOT}/bin:$PATH
    export LD_LIBRARY_PATH=${CLANG_ROOT}/lib:$LD_LIBRARY_PATH
    #
    export CFLAGS=\"${CFLAGS}\"
    export CXXFLAGS=\"${CXXFLAGS}\"
    export LDFLAGS=\"${LDFLAGS}\"
    export LDCXXFLAGS=\"${LDCXXFLAGS}\"
    #
    export CC=${CLANG_ROOT}/bin/clang
    export CXX=${CLANG_ROOT}/bin/clang++
    export CPP=${CLANG_ROOT}/bin/clang-cpp
    #
  ")

  string(CONCAT CTEST_BUILD_OPTIONS
    "  -DHPX_WITH_CXX17=ON "
    "  -DBoost_COMPILER=-clang60 "
  )
endif()

set(HWLOC_ROOT       "${INSTALL_ROOT}/hwloc/${HWLOC_VER}")
set(JEMALLOC_ROOT    "${INSTALL_ROOT}/jemalloc/${JEMALLOC_VER}")
set(OTF2_ROOT        "${INSTALL_ROOT}/otf2/${OTF2_VER}")
set(PAPI_ROOT        "${INSTALL_ROOT}/papi/${PAPI_VER}")
set(PAPI_INCLUDE_DIR "${INSTALL_ROOT}/papi/${PAPI_VER}/include")
set(PAPI_LIBRARY     "${INSTALL_ROOT}/papi/${PAPI_VER}/lib/libpfm.so")

set(CTEST_SITE "cray(daint)-${PYCICLE_BUILD_STAMP}-Boost-${BOOST_VER}")
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_TEST_TIMEOUT "45")
set(BUILD_PARALLELISM  "32")

#######################################################################
# The string that is used to drive cmake config step
# ensure options (e.g.FLAGS) that have multiple args are escaped
#######################################################################
string(CONCAT CTEST_BUILD_OPTIONS ${CTEST_BUILD_OPTIONS}
    "\"-DCMAKE_CXX_FLAGS=${CXXFLAGS}\" "
    "\"-DCMAKE_C_FLAGS=${CFLAGS}\" "
    "\"-DCMAKE_EXE_LINKER_FLAGS=${LDCXXFLAGS}\" "
    "  -DHPX_WITH_NATIVE_TLS=ON "
    "  -DHWLOC_ROOT=${HWLOC_ROOT} "
    "  -DJEMALLOC_ROOT=${JEMALLOC_ROOT} "
    "  -DBOOST_ROOT=${BOOST_ROOT} "
    "  -DBoost_ADDITIONAL_VERSIONS=${BOOST_VER} "
    "  -DHPX_WITH_MALLOC=JEMALLOC "
    "  -DHPX_WITH_EXAMPLES=ON "
    "  -DHPX_WITH_TESTS=ON "
    "  -DHPX_WITH_TESTS_BENCHMARKS=ON "
    "  -DHPX_WITH_TESTS_EXTERNAL_BUILD=OFF "
    "  -DHPX_WITH_TESTS_HEADERS=OFF "
    "  -DHPX_WITH_TESTS_REGRESSIONS=ON "
    "  -DHPX_WITH_TESTS_UNIT=ON "
    "  -DHPX_WITH_PARCELPORT_MPI=OFF "
    "  -DHPX_WITH_PARCELPORT_MPI_MULTITHREADED=OFF "
    "  -DHPX_WITH_THREAD_IDLE_RATES=ON "
    "  -DHPX_WITH_MAX_CPU_COUNT=256 "
    "  -DHPX_WITH_MORE_THAN_64_THREADS=ON "
)

#######################################################################
# Setup a slurm job submission template
# note that this is intentionally multiline
#######################################################################
set(PYCICLE_SLURM_TEMPLATE "#!/bin/bash
#SBATCH --job-name=hpx-${PYCICLE_PR}-${PYCICLE_BUILD_STAMP}
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=mc
#SBATCH --partition=normal

# ---------------------
# unload or load modules that differ from the defaults on the system
# ---------------------
module load   slurm
module load   git
module load   CMake/${CMAKE_VER}
module unload gcc

#
# ---------------------
# setup stuff that might differ between compilers
# ---------------------
${PYCICLE_COMPILER_SETUP}

# ---------------------
# This is used by the hpx test runner
# ---------------------
export HPXRUN_RUNWRAPPER=srun
"
)
