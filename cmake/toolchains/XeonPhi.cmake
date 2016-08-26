# Copyright (c) 2014 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#
# This is the default toolchain file to be used with Intel Xeon PHIs. It sets
# the appropriate compile flags and compiler such that HPX will compile.
# Note that you still need to provide Boost, hwloc and other utility libraries
# like a custom allocator yourself.
#

set(CMAKE_SYSTEM_NAME Linux)

# Set the Intel Compiler
set(CMAKE_CXX_COMPILER icpc)
set(CMAKE_C_COMPILER icc)
set(CMAKE_Fortran_COMPILER ifort)

# Add the -mmic compile flag such that everything will be compiled for the correct
# platform
set(CMAKE_CXX_FLAGS_INIT "-mmic" CACHE STRING "Initial compiler flags used to compile for the Xeon Phi")
set(CMAKE_C_FLAGS_INIT "-mmic" CACHE STRING "Initial compiler flags used to compile for the Xeon Phi")
set(CMAKE_Fortran_FLAGS_INIT "-mmic" CACHE STRING "Initial compiler flags used to compile for the Xeon Phi")

# Disable searches in the default system paths. We are cross compiling after all
# and cmake might pick up wrong libraries that way
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# We do a cross compilation here ...
set(CMAKE_CROSSCOMPILING ON)

# Set our platform name
set(HPX_PLATFORM "XeonPhi")

# Always disable the ibverbs parcelport as it is non-functional on the BGQ.
set(HPX_WITH_PARCELPORT_IBVERBS OFF CACHE BOOL "Enable the ibverbs based parcelport. This is currently an experimental feature")

# We have a bunch of cores on the MIC ... increase the default
set(HPX_WITH_MAX_CPU_COUNT "256" CACHE STRING "")

# We default to tbbmalloc as our allocator on the MIC
if(NOT DEFINED HPX_WITH_MALLOC)
  set(HPX_WITH_MALLOC "tbbmalloc" CACHE STRING "")
endif()

# Set the TBBMALLOC_PLATFORM correctly so that find_package(TBBMalloc) sets the
# right hints
set(TBBMALLOC_PLATFORM "mic" CACHE STRING "")

set(HPX_HIDDEN_VISIBILITY OFF CACHE BOOL "Use -fvisibility=hidden for builds on platforms which support it")

# RDTSC is available on Xeon/Phis
set(HPX_WITH_RDTSC ON CACHE BOOL "")

