# Copyright (c) 2014 John Biddiscombe
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# This is the default toolchain file to be used with CNK on a BlueGene/Q. It sets
# the appropriate compile flags and compiler such that HPX will compile.
# Note that you still need to provide Boost, hwloc and other utility libraries
# like a custom allocator yourself.

#
# Usage : cmake -DCMAKE_TOOLCHAIN_FILE=~/src/hpx/cmake/toolchains/BGION-gcc.cmake ~/src/hpx
#

set(CMAKE_SYSTEM_NAME Linux)

# Set the gcc Compiler
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_C_COMPILER gcc)
#set(CMAKE_Fortran_COMPILER)

# Add flags we need for BGAS compilation
set(CMAKE_CXX_FLAGS_INIT
  "-D__powerpc__ -D__bgion__ -I/gpfs/bbp.cscs.ch/home/biddisco/src/bgas/rdmahelper "
  CACHE STRING "Initial compiler flags used to compile for BGAS"
)
# the V1R2M2 includes are necessary for some hardware specific features
#-DHPX_SMALL_STACK_SIZE=0x200000 -DHPX_MEDIUM_STACK_SIZE=0x200000 -DHPX_LARGE_STACK_SIZE=0x200000 -DHPX_HUGE_STACK_SIZE=0x200000

set(CMAKE_EXE_LINKER_FLAGS_INIT "-L/gpfs/bbp.cscs.ch/apps/bgas/tools/gcc/gcc-4.8.2/install/lib64 -latomic -lrt" CACHE STRING "BGAS flags")

set(CMAKE_C_FLAGS_INIT "-D__powerpc__ -I/gpfs/bbp.cscs.ch/home/biddisco/src/bgas/rdmahelper" CACHE STRING "BGAS flags")

# We do not perform cross compilation here ...
set(CMAKE_CROSSCOMPILING OFF)

# Set our platform name
set(HPX_PLATFORM "native")

# Disable generic coroutines (and use posix version)
set(HPX_WITH_GENERIC_CONTEXT_COROUTINES OFF CACHE BOOL "diable generic coroutines")

# BGAS nodes support ibverbs
set(HPX_WITH_PARCELPORT_IBVERBS ON CACHE BOOL "")

# Always disable the tcp parcelport as it is nonfunctional on the BGQ.
set(HPX_WITH_PARCELPORT_TCP ON CACHE BOOL "")

# Always enable the tcp parcelport as it is currently the only way to communicate on the BGQ.
set(HPX_WITH_PARCELPORT_MPI ON CACHE BOOL "")

# We have a bunch of cores on the A2 processor ...
set(HPX_WITH_MAX_CPU_COUNT "64" CACHE STRING "")

# We have no custom malloc yet
if(NOT DEFINED HPX_WITH_MALLOC)
  set(HPX_WITH_MALLOC "system" CACHE STRING "")
endif()

set(HPX_HIDDEN_VISIBILITY OFF CACHE BOOL "")

#
# Convenience setup for jb @ bbpbg2.cscs.ch
#
set(BOOST_ROOT "/gpfs/bbp.cscs.ch/home/biddisco/apps/gcc-4.8.2/boost_1_56_0")
set(HWLOC_ROOT "/gpfs/bbp.cscs.ch/home/biddisco/apps/gcc-4.8.2/hwloc-1.8.1")
set(HPX_WITH_HWLOC ON CACHE BOOL "Use hwloc")

set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Default build")

#
# Testing flags
#
set(BUILD_TESTING                  ON  CACHE BOOL "Testing enabled by default")
set(HPX_WITH_TESTS                ON  CACHE BOOL "Testing enabled by default")
set(HPX_WITH_TESTS_BENCHMARKS     ON  CACHE BOOL "Testing enabled by default")
set(HPX_WITH_TESTS_REGRESSIONS    ON  CACHE BOOL "Testing enabled by default")
set(HPX_WITH_TESTS_UNIT           ON  CACHE BOOL "Testing enabled by default")
set(HPX_WITH_TESTS_EXTERNAL_BUILD OFF CACHE BOOL "Turn off build of cmake build tests")
set(DART_TESTING_TIMEOUT           45  CACHE STRING "Life is too short")

# HPX_WITH_STATIC_LINKING
