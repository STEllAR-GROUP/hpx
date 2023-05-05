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
   pkg_check_modules(PC_GASNET QUIET gasnet-mpi-par)

   find_path(
      GASNET_INCLUDE_DIR gasnet.h
      HINTS ${GASNET_ROOT} ENV GASNET_ROOT ${PC_GASNET_INCLUDEDIR}
            ${PC_GASNET_INCLUDE_DIRS} 
      PATH_SUFFIXES include
   )

   find_library(
      GASNET_LIBRARY
      NAMES gasnet-mpi-par
      HINTS ${GASNET_ROOT} ENV GASNET_ROOT ${PC_GASNET_LIBDIR}
            ${PC_GASNET_LIBRARY_DIRS}
      PATH_SUFFIXES lib lib64
   )

   set(GASNET_MPI_FOUND TRUE)
   hpx_setup_mpi()
endif()

if(HPX_WITH_PARCELPORT_GASNET_UDP)
   pkg_check_modules(GASNET QUIET gasnet-udp-par)

   find_path(
      GASNET_INCLUDE_DIR gasnet.h
      HINTS ${GASNET_ROOT} ENV GASNET_ROOT ${PC_GASNET_INCLUDEDIR}
            ${PC_GASNET_INCLUDE_DIRS} 
      PATH_SUFFIXES include
   )

   find_library(
      GASNET_LIBRARY
      NAMES gasnet-udp-par
      HINTS ${GASNET_ROOT} ENV GASNET_ROOT ${PC_GASNET_LIBDIR}
            ${PC_GASNET_LIBRARY_DIRS}
      PATH_SUFFIXES lib lib64
   )

   set(GASNET_UDP_FOUND TRUE)
endif()

if(HPX_WITH_PARCELPORT_GASNET_SMP)
   pkg_check_modules(GASNET QUIET gasnet-smp-par)

   find_path(
      GASNET_INCLUDE_DIR gasnet.h
      HINTS ${GASNET_ROOT} ENV GASNET_ROOT ${PC_GASNET_INCLUDEDIR}
            ${PC_GASNET_INCLUDE_DIRS} 
      PATH_SUFFIXES include
   )

   find_library(
      GASNET_LIBRARY
      NAMES gasnet-smp-par
      HINTS ${GASNET_ROOT} ENV GASNET_ROOT ${PC_GASNET_LIBDIR}
            ${PC_GASNET_LIBRARY_DIRS}
      PATH_SUFFIXES lib lib64
   )

   set(GASNET_SMP_FOUND TRUE)
endif()


# Set GASNET_ROOT in case the other hints are used
if(NOT GASNET_ROOT AND "$ENV{GASNET_ROOT}")
  set(GASNET_ROOT $ENV{GASNET_ROOT})
elseif(NOT GASNET_ROOT)
  string(REPLACE "/include" "" GASNET_ROOT "${GASNET_INCLUDE_DIR}")
endif()

# Set GASNET_ROOT in case the other hints are used
if(GASNET_ROOT)
  # The call to file is for compatibility with windows paths
  file(TO_CMAKE_PATH ${GASNET_ROOT} GASNET_ROOT)
elseif("$ENV{GASNET_ROOT}")
  file(TO_CMAKE_PATH $ENV{GASNET_ROOT} GASNET_ROOT)
else()
  file(TO_CMAKE_PATH "${GASNET_INCLUDE_DIR}" GASNET_INCLUDE_DIR)
  string(REPLACE "/include" "" GASNET_ROOT "${GASNET_INCLUDE_DIR}")
endif()

set(GASNET_LIBRARIES ${GASNET_LIBRARY})

if(HPX_WITH_PARCELPORT_GASNET_MPI)
  set(GASNET_CONDUIT_INCLUDE_DIR "${GASNET_INCLUDE_DIR}/mpi-conduit")
elseif(HPX_WITH_PARCELPORT_GASNET_UDP)
  set(GASNET_CONDUIT_INCLUDE_DIR "${GASNET_INCLUDE_DIR}/udp-conduit")
elseif(HPX_WITH_PARCELPORT_GASNET_SMP)
  set(GASNET_CONDUIT_INCLUDE_DIR "${GASNET_INCLUDE_DIR}/smp-conduit")
endif()

set(GASNET_INCLUDE_DIRS ${GASNET_INCLUDE_DIR} ${GASNET_CONDUIT_INCLUDE_DIR})

find_package_handle_standard_args(
  GASNET DEFAULT_MSG GASNET_LIBRARY GASNET_INCLUDE_DIR
)

get_property(
  _type
  CACHE GASNET_ROOT
  PROPERTY TYPE
)

if(_type)
  set_property(CACHE GASNET_ROOT PROPERTY ADVANCED 1)
  if("x${_type}" STREQUAL "xUNINITIALIZED")
    set_property(CACHE GASNET_ROOT PROPERTY TYPE PATH)
  endif()
endif()

mark_as_advanced(GASNET_ROOT GASNET_LIBRARY GASNET_INCLUDE_DIR GASNET_CONDUIT_INCLUDE_DIR)

add_library(Gasnet::gasnet INTERFACE IMPORTED)
target_include_directories(Gasnet::gasnet SYSTEM INTERFACE "${GASNET_INCLUDE_DIR}" "${GASNET_CONDUIT_INCLUDE_DIR}")
target_link_libraries(Gasnet::gasnet INTERFACE ${GASNET_LIBRARY})

endmacro()
