# Copyright (c) 2014-2017 Thomas Heller
# Copyright (c) 2017      Bryce Adelstein Lelbach 
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_WITH_STATIC_LINKING ON CACHE BOOL "")
set(HPX_WITH_STATIC_EXE_LINKING ON CACHE BOOL "")
set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS FALSE)

# Set the Cray Compiler Wrapper
set(CMAKE_CXX_COMPILER CC)
set(CMAKE_C_COMPILER cc)
set(CMAKE_Fortran_COMPILER ftn)

if (CMAKE_VERSION VERSION_GREATER 3.3.9)
  set(__includes "<INCLUDES>")
endif()

set(CMAKE_C_FLAGS_INIT "" CACHE STRING "")
set(CMAKE_C_COMPILE_OBJECT "<CMAKE_C_COMPILER> -static -fPIC <DEFINES> ${__includes} <FLAGS> -o <OBJECT> -c <SOURCE>" CACHE STRING "")
set(CMAKE_C_LINK_EXECUTABLE "<CMAKE_C_COMPILER> -fPIC <FLAGS> <CMAKE_C_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>" CACHE STRING "")

set(CMAKE_CXX_FLAGS_INIT "" CACHE STRING "")
set(CMAKE_CXX_COMPILE_OBJECT "<CMAKE_CXX_COMPILER> -static -fPIC <DEFINES> ${__includes} <FLAGS> -o <OBJECT> -c <SOURCE>" CACHE STRING "")
set(CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_CXX_COMPILER> -fPIC <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>" CACHE STRING "")

set(CMAKE_Fortran_FLAGS_INIT "" CACHE STRING "")
set(CMAKE_Fortran_COMPILE_OBJECT "<CMAKE_Fortran_COMPILER> -static -fPIC <DEFINES> ${__includes} <FLAGS> -o <OBJECT> -c <SOURCE>" CACHE STRING "")
set(CMAKE_Fortran_LINK_EXECUTABLE "<CMAKE_Fortran_COMPILER> -fPIC <FLAGS> <CMAKE_Fortran_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")

# Disable searches in the default system paths. We are cross compiling after all
# and cmake might pick up wrong libraries that way
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(HPX_WITH_PARCELPORT_TCP ON CACHE BOOL "")

set(HPX_WITH_PARCELPORT_MPI ON CACHE BOOL "")
set(HPX_WITH_PARCELPORT_MPI_MULTITHREADED ON CACHE BOOL "")

set(HPX_WITH_PARCELPORT_LIBFABRIC ON CACHE BOOL "")
set(HPX_PARCELPORT_LIBFABRIC_PROVIDER "gni" CACHE STRING
  "See libfabric docs for details, gni,verbs,psm2 etc etc")
set(HPX_PARCELPORT_LIBFABRIC_THROTTLE_SENDS "256" CACHE STRING
  "Max number of messages in flight at once")
set(HPX_PARCELPORT_LIBFABRIC_WITH_DEV_MODE OFF CACHE BOOL
  "Custom libfabric logging flag")
set(HPX_PARCELPORT_LIBFABRIC_WITH_LOGGING  OFF CACHE BOOL
  "Libfabric parcelport logging on/off flag")
set(HPX_WITH_ZERO_COPY_SERIALIZATION_THRESHOLD "4096" CACHE STRING
  "The threshhold in bytes to when perform zero copy optimizations (default: 128)")

# Set the TBBMALLOC_PLATFORM correctly so that find_package(TBBMalloc) sets the
# right hints
set(TBBMALLOC_PLATFORM "mic-knl" CACHE STRING "")

# We have a bunch of cores on the MIC ... increase the default
set(HPX_WITH_MAX_CPU_COUNT "512" CACHE STRING "")

# We do a cross compilation here ...
set(CMAKE_CROSSCOMPILING ON CACHE BOOL "")

# RDTSCP is available on Xeon/Phis
set(HPX_WITH_RDTSCP ON CACHE BOOL "")
