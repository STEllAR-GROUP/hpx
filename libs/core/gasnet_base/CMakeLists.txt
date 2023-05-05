# Copyright (c) 2019-2023 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(NOT (HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_GASNET AND (HPX_WITH_PARCELPORT_GASNET_MPI OR HPX_WITH_PARCELPORT_GASNET_SMP OR HPX_WITH_PARCELPORT_GASNET_UDP)))
  message(STATUS "*** PARCELPORT_GASNET not set")
  return()
endif()

# setup GASNET, if necessary
include(HPX_SetupGASNET)
hpx_setup_gasnet()

include(HPX_AddModule)

# Default location is $HPX_ROOT/libs/gasnet_base/include
set(gasnet_base_headers hpx/gasnet_base/gasnet.hpp hpx/gasnet_base/gasnet_environment.hpp)

set(gasnet_base_sources gasnet_environment.cpp)

if(GASNET_MPI_FOUND AND HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_MPI)
  set(additional_dependencies Mpi::mpi Gasnet::gasnet)

  add_hpx_module(
    core gasnet_base
    GLOBAL_HEADER_GEN ON
    SOURCES ${gasnet_base_sources}
    HEADERS ${gasnet_base_headers}
    MODULE_DEPENDENCIES hpx_logging hpx_runtime_configuration hpx_util
    DEPENDENCIES ${additional_dependencies}
    CMAKE_SUBDIRS examples tests
  )
else()
  add_hpx_module(
    core gasnet_base
    GLOBAL_HEADER_GEN ON
    SOURCES ${gasnet_base_sources}
    HEADERS ${gasnet_base_headers}
    MODULE_DEPENDENCIES hpx_logging hpx_runtime_configuration hpx_util
    DEPENDENCIES Gasnet::gasnet
    CMAKE_SUBDIRS examples tests
  )
endif()
