# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

set(debugging_headers
  hpx/debugging/demangle_helper.hpp
  # do not add hpx/debugging/print.hpp as it includes threads
  # instead users should add this manually on demand
  )

set(debugging_compat_headers
  hpx/util/debug/demangle_helper.hpp
  )

set(debugging_sources
  )

include(HPX_AddModule)
add_hpx_module(debugging
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  FORCE_LINKING_GEN
  GLOBAL_HEADER_GEN ON
  SOURCES ${debugging_sources}
  HEADERS ${debugging_headers}
  COMPAT_HEADERS ${debugging_compat_headers}
  DEPENDENCIES
    hpx_config
    hpx_errors
    hpx_functional
    hpx_coroutines
    hpx_hardware
    hpx_topology
    hpx_program_options
  CMAKE_SUBDIRS examples tests
)
