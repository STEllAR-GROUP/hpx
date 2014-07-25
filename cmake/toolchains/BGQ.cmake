# Copyright (c) 2014 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# This is the default toolchain file to be used with CNK on a BlueGene/Q. It sets
# the appropriate compile flags and compiler such that HPX will compile.
# Note that you still need to provide Boost, hwloc and other utility libraries
# like a custom allocator yourself.

set(CMAKE_SYSTEM_NAME Linux)

# Set the Intel Compiler
set(CMAKE_CXX_COMPILER bgclang++11)
set(CMAKE_C_COMPILER bgclang)
#set(CMAKE_Fortran_COMPILER)

# Add the -mmic compile flag such that everything will be compiled for the correct
# platform
set(CMAKE_CXX_FLAGS_INIT "-mmic" CACHE STRING "Initial compiler flags used to compile for the Bluegene/Q")
set(CMAKE_C_FLAGS_INIT "-mmic" CACHE STRING "Initial compiler flags used to compile for the Bluegene/Q")
set(CMAKE_Fortran_FLAGS_INIT "-mmic" CACHE STRING "Initial compiler flags used to compile for the Bluegene/Q")

# Disable searches in the default system paths. We are cross compiling after all
# and cmake might pick up wrong libraries that way
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# We do a cross compilation here ...
set(CMAKE_CROSSCOMPILING ON)

# Set our platform name
set(HPX_PLATFORM "BlueGeneQ")

# Always disable the ibverbs parcelport as it is nonfunctional on the BGQ.
set(WITH_IBVERBS_PARCELPORT OFF)

# Always disable the tcp parcelport as it is nonfunctional on the BGQ.
set(WITH_TCP_PARCELPORT OFF)

# Always enable the tcp parcelport as it is currently the only way to communicate on the BGQ.
set(WITH_MPI_PARCELPORT OFF)

# We have a bunch of cores on the BGQ ...
set(HPX_MAX_CPU_COUNT "64")

# We default to tbbmalloc as our allocator on the MIC
if(NOT DEFINED HPX_MALLOC)
  set(HPX_MALLOC "system")
endif()
