# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

# if the user has not enabled libCDS, exit immediately
if(NOT ${HPX_WITH_LIBCDS})
  return()
endif()

include(HPX_SetupLibCDS)

if(NOT libcds_POPULATED)
  hpx_error("HPX_WITH_LIBCDS was set to ON, but HPX failed to fetch LibCDS")
endif()

# Default location is $HPX_ROOT/libs/libcds/include
set(libcds_headers hpx/libcds/hpx_tls_manager.hpp)

# Default location is $HPX_ROOT/libs/libcds/src
set(libcds_sources hpx_tls_manager.cpp)

include(HPX_AddModule)
add_hpx_module(
  libcds
  GLOBAL_HEADER_GEN ON
  SOURCES ${libcds_sources}
  HEADERS ${libcds_headers}
  DEPENDENCIES
    hpx_assertion
    hpx_execution_base
    hpx_concurrency
    hpx_config
    hpx_errors
    hpx_execution
    hpx_lcos_local
    hpx_memory
    hpx_threading_base
    hpx_threading
    libcds::cds
  CMAKE_SUBDIRS examples tests
)

target_include_directories(hpx_libcds PRIVATE ${LIBCDS_SOURCE_DIR})

# not needed with HP template specialization used in current module but would be
# needed if using another branch with more HPX specific code
# target_compile_definitions(hpx_libcds PUBLIC CDS_THREADING_HPX)
# target_compile_definitions(hpx_libcds PUBLIC CDS_EXPORT_API=HPX_EXPORT)

target_compile_definitions(hpx_libcds PUBLIC CDS_THREADING_CXX11)
