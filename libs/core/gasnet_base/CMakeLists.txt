# Copyright (c) 2019-2023 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT (HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_GASNET AND (HPX_WITH_PARCELPORT_GASNET_MPI OR HPX_WITH_PARCELPORT_GASNET_SMP OR HPX_WITH_PARCELPORT_GASNET_UDP OR HPX_WITH_PARCELPORT_GASNET_OFI OR HPX_WITH_PARCELPORT_GASNET_UCX OR HPX_WITH_PARCELPORT_GASNET_IBV)))
  message(STATUS "*** PARCELPORT_GASNET not set")
  return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(FindGasnet)
find_gasnet()

# Default location is $HPX_ROOT/libs/gasnet_base/include
set(gasnet_base_headers
   hpx/gasnet_base/gasnet.hpp
   hpx/gasnet_base/gasnet_environment.hpp
)

# Default location is $HPX_ROOT/libs/gasnet_base/include
# cmake-format: off
set(gasnet_base_compat_headers
    hpx/plugins/parcelport/gasnet/gasnet.hpp => hpx/modules/gasnet_base.hpp
    hpx/plugins/parcelport/gasnet/gasnet_environment.hpp => hpx/modules/gasnet_base.hpp
)
# cmake-format: on

set(gasnet_base_sources gasnet_environment.cpp)

if(GASNET_MPI_FOUND AND HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_MPI)
  include(HPX_AddModule)
  add_hpx_module(
    core gasnet_base
    GLOBAL_HEADER_GEN ON
    SOURCES ${gasnet_base_sources}
    HEADERS ${gasnet_base_headers}
    COMPAT_HEADERS ${gasnet_base_compat_headers}
    DEPENDENCIES PkgConfig::GASNET Mpi::mpi
    MODULE_DEPENDENCIES hpx_logging hpx_runtime_configuration hpx_util
    CMAKE_SUBDIRS examples tests
  )
else()
  include(HPX_AddModule)
  add_hpx_module(
    core gasnet_base
    GLOBAL_HEADER_GEN ON
    SOURCES ${gasnet_base_sources}
    HEADERS ${gasnet_base_headers}
    COMPAT_HEADERS ${gasnet_base_compat_headers}
    DEPENDENCIES PkgConfig::GASNET
    MODULE_DEPENDENCIES hpx_logging hpx_runtime_configuration hpx_util
    CMAKE_SUBDIRS examples tests
  )
endif()
