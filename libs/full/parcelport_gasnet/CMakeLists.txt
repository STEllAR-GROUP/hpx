# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT (HPX_WITH_NETWORKING AND (HPX_WITH_PARCELPORT_GASNET_MPI OR HPX_WITH_PARCELPORT_GASNET_SMP OR HPX_WITH_PARCELPORT_UDP OR HPX_WITH_PARCELPORT_OFI OR HPX_WITH_PARCELPORT_UCX OR HPX_WITH_PARCELPORT_IBV)))
  return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FindGasnet)
find_gasnet()

set(parcelport_gasnet_headers
    hpx/parcelport_gasnet/header.hpp
    hpx/parcelport_gasnet/locality.hpp
    hpx/parcelport_gasnet/receiver.hpp
    hpx/parcelport_gasnet/sender.hpp
    hpx/parcelport_gasnet/sender_connection.hpp
)

# cmake-format: off
set(parcelport_gasnet_compat_headers)
# cmake-format: on

set(parcelport_gasnet_sources locality.cpp parcelport_gasnet.cpp)

if(GASNET_MPI_FOUND AND HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_MPI)
link_directories(${GASNET_LIBRARY_DIRS})
   include(HPX_AddModule)
   add_hpx_module(
      full parcelport_gasnet
      GLOBAL_HEADER_GEN ON
      SOURCES ${parcelport_gasnet_sources}
      HEADERS ${parcelport_gasnet_headers}
      COMPAT_HEADERS ${parcelport_gasnet_compat_headers}
      DEPENDENCIES hpx_core hpx_gasnet_base PkgConfig::GASNET Mpi::mpi 
      MODULE_DEPENDENCIES hpx_actions hpx_command_line_handling hpx_parcelset
      CMAKE_SUBDIRS examples tests
   )

elseif(GASNET_UDP_FOUND AND HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_UDP)
link_directories(${GASNET_LIBRARY_DIRS})
   include(HPX_AddModule)
   add_hpx_module(
      full parcelport_gasnet
      GLOBAL_HEADER_GEN ON
      SOURCES ${parcelport_gasnet_sources}
      HEADERS ${parcelport_gasnet_headers}
      COMPAT_HEADERS ${parcelport_gasnet_compat_headers}
      DEPENDENCIES hpx_core hpx_gasnet_base PkgConfig::GASNET
      MODULE_DEPENDENCIES hpx_actions hpx_command_line_handling hpx_parcelset
      CMAKE_SUBDIRS examples tests
   )

elseif(GASNET_OFI_FOUND AND HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_OFI)
link_directories(${GASNET_LIBRARY_DIRS})
   include(HPX_AddModule)
   add_hpx_module(
      full parcelport_gasnet
      GLOBAL_HEADER_GEN ON
      SOURCES ${parcelport_gasnet_sources}
      HEADERS ${parcelport_gasnet_headers}
      COMPAT_HEADERS ${parcelport_gasnet_compat_headers}
      DEPENDENCIES hpx_core hpx_gasnet_base PkgConfig::GASNET
      MODULE_DEPENDENCIES hpx_actions hpx_command_line_handling hpx_parcelset
      CMAKE_SUBDIRS examples tests
   )

elseif(GASNET_UCX_FOUND AND HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_UCX)
link_directories(${GASNET_LIBRARY_DIRS})
   include(HPX_AddModule)
   add_hpx_module(
      full parcelport_gasnet
      GLOBAL_HEADER_GEN ON
      SOURCES ${parcelport_gasnet_sources}
      HEADERS ${parcelport_gasnet_headers}
      COMPAT_HEADERS ${parcelport_gasnet_compat_headers}
      DEPENDENCIES hpx_core hpx_gasnet_base PkgConfig::GASNET
      MODULE_DEPENDENCIES hpx_actions hpx_command_line_handling hpx_parcelset
      CMAKE_SUBDIRS examples tests
   )

elseif(GASNET_IBV_FOUND AND HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_IBV)
link_directories(${GASNET_LIBRARY_DIRS})
   include(HPX_AddModule)
   add_hpx_module(
      full parcelport_gasnet
      GLOBAL_HEADER_GEN ON
      SOURCES ${parcelport_gasnet_sources}
      HEADERS ${parcelport_gasnet_headers}
      COMPAT_HEADERS ${parcelport_gasnet_compat_headers}
      DEPENDENCIES hpx_core hpx_gasnet_base PkgConfig::GASNET
      MODULE_DEPENDENCIES hpx_actions hpx_command_line_handling hpx_parcelset
      CMAKE_SUBDIRS examples tests
   )

elseif(GASNET_SMP_FOUND)
link_directories(${GASNET_LIBRARY_DIRS})
   include(HPX_AddModule)
   add_hpx_module(
      full parcelport_gasnet
      GLOBAL_HEADER_GEN ON
      SOURCES ${parcelport_gasnet_sources}
      HEADERS ${parcelport_gasnet_headers}
      COMPAT_HEADERS ${parcelport_gasnet_compat_headers}
      DEPENDENCIES hpx_core hpx_gasnet_base PkgConfig::GASNET
      MODULE_DEPENDENCIES hpx_actions hpx_command_line_handling hpx_parcelset
      CMAKE_SUBDIRS examples tests
   )
endif()

set(HPX_STATIC_PARCELPORT_PLUGINS
    ${HPX_STATIC_PARCELPORT_PLUGINS} parcelport_gasnet
    CACHE INTERNAL "" FORCE
)
