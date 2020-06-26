# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(util_headers
    hpx/util/detail/reserve.hpp
    hpx/util/calculate_fanout.hpp
    hpx/util/get_and_reset_value.hpp
    hpx/util/get_entry_as.hpp
    hpx/util/insert_checked.hpp
    hpx/util/ios_flags_saver.hpp
    hpx/util/manage_config.hpp
    hpx/util/regex_from_pattern.hpp
    hpx/util/sed_transform.hpp
    hpx/util/traits/await_traits.hpp
)

set(util_compat_headers hpx/util.hpp hpx/util/await_traits.hpp)

set(util_sources manage_config.cpp regex_from_pattern.cpp sed_transform.cpp)

include(HPX_AddModule)
add_hpx_module(
  core util
  COMPATIBILITY_HEADERS ON # Added in 1.5.0
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${util_sources}
  HEADERS ${util_headers}
  COMPAT_HEADERS ${util_compat_headers}
  MODULE_DEPENDENCIES hpx_config hpx_concepts hpx_errors hpx_format
                      hpx_iterator_support
  CMAKE_SUBDIRS examples tests
)
