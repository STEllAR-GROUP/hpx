# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/timing/include
set(timing_headers
    hpx/timing/high_resolution_clock.hpp hpx/timing/high_resolution_timer.hpp
    hpx/timing/scoped_timer.hpp hpx/timing/steady_clock.hpp
    hpx/timing/tick_counter.hpp
)

# Default location is $HPX_ROOT/libs/timing/include_compatibility
set(timing_compat_headers
    hpx/timing.hpp hpx/util/high_resolution_clock.hpp
    hpx/util/high_resolution_timer.hpp hpx/util/scoped_timer.hpp
    hpx/util/steady_clock.hpp hpx/util/tick_counter.hpp
)

# cmake will not create a separate VS project without any source files, thus
# this adds a dummy (empty) source file to the target Default location is
# $HPX_ROOT/libs/timing/src
set(timing_sources)

include(HPX_AddModule)
add_hpx_module(
  core timing
  COMPATIBILITY_HEADERS ON # Added in 1.5.0
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${timing_sources}
  HEADERS ${timing_headers}
  COMPAT_HEADERS ${timing_compat_headers}
  MODULE_DEPENDENCIES hpx_config hpx_hardware hpx_type_support
  CMAKE_SUBDIRS examples tests
)
