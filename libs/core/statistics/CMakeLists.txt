# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/statistics/include
set(statistics_headers
    hpx/statistics/histogram.hpp hpx/statistics/max.hpp hpx/statistics/min.hpp
    hpx/statistics/rolling_max.hpp hpx/statistics/rolling_min.hpp
)

# Default location is $HPX_ROOT/libs/statistics/include_compatibility
# cmake-format: off
set(statistics_compat_headers
    hpx/statistics.hpp => hpx/modules/statistics.hpp
    hpx/util/histogram.hpp => hpx/modules/statistics.hpp
    hpx/util/max.hpp => hpx/modules/statistics.hpp
    hpx/util/min.hpp => hpx/modules/statistics.hpp
    hpx/util/rolling_max.hpp => hpx/modules/statistics.hpp
    hpx/util/rolling_min.hpp => hpx/modules/statistics.hpp
)
# cmake-format: on

# Default location is $HPX_ROOT/libs/statistics/src
set(statistics_sources)

include(HPX_AddModule)
add_hpx_module(
  core statistics
  COMPATIBILITY_HEADERS ON # Added in 1.5.0
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${statistics_sources}
  HEADERS ${statistics_headers}
  COMPAT_HEADERS ${statistics_compat_headers}
  MODULE_DEPENDENCIES hpx_config hpx_iterator_support
  CMAKE_SUBDIRS examples tests
)
