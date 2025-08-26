# Copyright (c) 2025 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(version_headers hpx/version.hpp hpx/version_api.hpp)

set(version_sources version.cpp)

if((HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_MPI) OR HPX_WITH_ASYNC_MPI)
  include(HPX_SetupMPI)
  hpx_setup_mpi()
  set(additional_dependencies ${additional_dependencies} Mpi::mpi)
endif()
if(HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_LCI)
  include(HPX_SetupLCI)
  hpx_setup_lci()
  set(additional_dependencies ${additional_dependencies} LCI::LCI)
endif()
if(HPX_WITH_NETWORKING
   AND HPX_WITH_PARCELPORT_GASNET
   AND HPX_WITH_PARCELPORT_GASNET_CONDUIT
)
  include(HPX_SetupGasnet)
  hpx_setup_gasnet()
  set(additional_dependencies ${additional_dependencies} PkgConfig::GASNET)
endif()

include(HPX_AddModule)
add_hpx_module(
  core version
  GLOBAL_HEADER_GEN ON
  GLOBAL_HEADER_MODULE_GEN ON
  SOURCES ${version_sources}
  HEADERS ${version_headers}
  EXCLUDE_FROM_GLOBAL_HEADER hpx/version.hpp
  MODULE_DEPENDENCIES hpx_config hpx_config_registry hpx_format hpx_prefix
  DEPENDENCIES Hwloc::hwloc ${additional_dependencies}
  MODULE_DEPENDENCIES hpx_config hpx_format hpx_prefix
  CMAKE_SUBDIRS tests
)

target_include_directories(
  hpx_version PRIVATE $<BUILD_INTERFACE:${HPX_BINARY_DIR}>
)
