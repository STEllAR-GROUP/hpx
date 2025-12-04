# Copyright (c) 2019-2025 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

hpx_option(
  HPX_LIKWID_WITH_LIKWID BOOL "Enable Likwid instrumentation support." OFF
  CATEGORY "Profiling"
)

# enable likwid module only if HPX_LIKWID_WITH_LIKWID is set
if(NOT ${HPX_LIKWID_WITH_LIKWID})
  return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Setup likwid
if(HPX_LIKWID_WITH_LIKWID)
  include(HPX_SetupLikwid)
endif()

set(likwid_headers hpx/likwid/likwid_tls.hpp)

set(likwid_sources likwid_tls.cpp)

include(HPX_AddModule)
add_hpx_module(
  core likwid
  GLOBAL_HEADER_GEN ON
  GLOBAL_HEADER_MODULE_GEN ON
  SOURCES ${likwid_sources}
  HEADERS ${likwid_headers}
  MODULE_DEPENDENCIES hpx_config
  DEPENDENCIES LIKWID::likwid
  CMAKE_SUBDIRS examples tests
)
