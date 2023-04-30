# Copyright (c)      2017 Thomas Heller
# Copyright (c)      2023 Christopher Taylor
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

macro(find_gasnet)
set(GASNET_MPI_FOUND FALSE)
set(GASNET_UDP_FOUND FALSE)
set(GASNET_SMP_FOUND FALSE)

find_package(PkgConfig QUIET)

if(HPX_WITH_PARCELPORT_GASNET_MPI)
   pkg_check_modules(GASNET QUIET gasnet-mpi-par)
   if(NOT GASNET_FOUND)
      hpx_error("Could not find GASNET MPI please set the PKG_CONFIG_PATH environment variable")
   endif()

   find_path(
      GASNET_INCLUDE_DIR gasnet.h
      HINTS ${GASNET_ROOT} ENV GASNETC_ROOT ${GASNET_DIR} ENV GASNET_DIR
      PATH_SUFFIXES include
   )

   find_library(
      GASNET_LIBRARY
      NAMES gasnet-mpi-par
      HINTS ${GASNET_ROOT} ENV GASNET_ROOT
      PATH_SUFFIXES lib lib64
   )

   set(GASNET_MPI_FOUND TRUE)
   hpx_setup_mpi()
endif()

if(HPX_WITH_PARCELPORT_GASNET_UDP)
   pkg_check_modules(GASNET QUIET gasnet-udp-par)
   if(NOT GASNET_FOUND)
      hpx_error("Could not find GASNET MPI/udp please set the PKG_CONFIG_PATH environment variable")
   endif()
   set(GASNET_UDP_FOUND TRUE)

   find_path(
      GASNET_INCLUDE_DIR gasnet.h
      HINTS ${GASNET_ROOT} ENV GASNETC_ROOT ${GASNET_DIR} ENV GASNET_DIR
      PATH_SUFFIXES include
   )

   find_library(
      GASNET_LIBRARY
      NAMES gasnet-udp-par
      HINTS ${GASNET_ROOT} ENV GASNET_ROOT
      PATH_SUFFIXES lib lib64
   )
endif()

if(HPX_WITH_PARCELPORT_GASNET_SMP)
   pkg_check_modules(GASNET QUIET gasnet-smp-par)
   if(NOT GASNET_FOUND)
      hpx_error("Could not find GASNET smp please set the PKG_CONFIG_PATH environment variable")
   endif()
   set(GASNET_SMP_FOUND TRUE)

   find_path(
      GASNET_INCLUDE_DIR gasnet.h
      HINTS ${GASNET_ROOT} ENV GASNETC_ROOT ${GASNET_DIR} ENV GASNET_DIR
      PATH_SUFFIXES include
   )

   find_library(
      GASNET_LIBRARY
      NAMES gasnet-smp-par
      HINTS ${GASNET_ROOT} ENV GASNET_ROOT
      PATH_SUFFIXES lib lib64
   )
endif()

set(GASNET_LIBRARIES ${GASNET_LIBRARY})
set(GASNET_INCLUDE_DIRS ${GASNET_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Gasnet DEFAULT_MSG GASNET_LIBRARY GASNET_INCLUDE_DIR
)

mark_as_advanced(GASNET_ROOT GASNET_LIBRARY GASNET_INCLUDE_DIR)
endmacro()
