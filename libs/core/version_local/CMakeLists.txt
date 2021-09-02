# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(version_local_headers hpx/local/version.hpp)

set(version_local_sources version.cpp)

if(HPXLocal_WITH_ASYNC_MPI)
  include(HPXLocal_SetupMPI)
  set(mpi_dependencies Mpi::mpi)
endif()

include(HPXLocal_AddModule)
hpx_local_add_module(
  local version_local
  GLOBAL_HEADER_GEN OFF
  SOURCES ${version_local_sources}
  HEADERS ${version_local_headers}
  MODULE_DEPENDENCIES hpx_config_local hpx_config_registry hpx_format hpx_prefix
  DEPENDENCIES Hwloc::hwloc ${mpi_dependencies}
  MODULE_DEPENDENCIES hpx_config_local hpx_format hpx_prefix
)

target_include_directories(
  hpx_version_local PRIVATE $<BUILD_INTERFACE:${HPXLocal_BINARY_DIR}>
)
