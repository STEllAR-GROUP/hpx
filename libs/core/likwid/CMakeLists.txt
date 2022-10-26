# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# enable likwid module only if HPX_WITH_LIKWID is set
if(NOT ${HPX_WITH_LIKWID})
  return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(likwid_headers
  hpx/likwid/likwid_init.hpp
  hpx/likwid/likwid_executor.hpp
)

set(likwid_sources
  likwid_init.cpp
)

set(execution_optional_dependencies LIKWID::likwid)

include(HPX_AddModule)
add_hpx_module(
  core likwid
  GLOBAL_HEADER_GEN ON
  SOURCES ${likwid_sources}
  HEADERS ${likwid_headers}
  DEPENDENCIES ${execution_optional_dependencies}
  MODULE_DEPENDENCIES
  hpx_execution
  hpx_runtime_local
  CMAKE_SUBDIRS examples tests
)
