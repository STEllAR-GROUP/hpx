# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# LibCDS option
hpx_option(
  HPX_LIBCDS_WITH_LIBCDS BOOL "Enable LibCDS support." OFF ADVANCED
  CATEGORY "Modules"
  MODULE LIBCDS
)
if(HPX_LIBCDS_WITH_LIBCDS)
  hpx_option(
    HPX_LIBCDS_WITH_GIT_REPOSITORY STRING
    "Define the LibCDS git repository to use."
    https://github.com/STEllAR-GROUP/libcds.git ADVANCED
    CATEGORY "Modules"
    MODULE LIBCDS
  )
  hpx_option(
    HPX_LIBCDS_WITH_GIT_TAG STRING "Define the LibCDS git tag to use."
    replace_boost_thread ADVANCED
    CATEGORY "Modules"
    MODULE LIBCDS
  )
endif()

# if the user has not enabled libCDS, exit immediately
if(NOT ${HPX_LIBCDS_WITH_LIBCDS})
  return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(HPX_SetupLibCDS)

if(NOT TARGET LibCDS::cds)
  hpx_error(
    "HPX_LIBCDS_WITH_LIBCDS was set to ON, but HPX failed to find LibCDS"
  )
endif()

# Default location is $HPX_ROOT/libs/libcds/include
set(libcds_headers hpx/libcds/cds_wrapper.hpp hpx/libcds/hpx_tls_manager.hpp
                   hpx/libcds/std_tls_manager.hpp
)

# Default location is $HPX_ROOT/libs/libcds/src
set(libcds_sources cds_wrapper.cpp hpx_tls_manager.cpp std_tls_manager.cpp)

include(HPX_AddModule)
add_hpx_module(
  parallelism libcds
  GLOBAL_HEADER_GEN ON
  SOURCES ${libcds_sources}
  HEADERS ${libcds_headers}
  MODULE_DEPENDENCIES hpx_execution hpx_lcos_local hpx_threading
  DEPENDENCIES hpx_core LibCDS::cds
  CMAKE_SUBDIRS examples tests
)

target_compile_definitions(hpx_libcds PUBLIC CDS_THREADING_HPX)
if(MSVC)
  target_compile_definitions(hpx_libcds PUBLIC NOMINMAX)
endif()
