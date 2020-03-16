# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(util_headers
  hpx/util/detail/reserve.hpp
  hpx/util/await_traits.hpp
  hpx/util/calculate_fanout.hpp
  hpx/util/get_and_reset_value.hpp
  hpx/util/insert_checked.hpp
  hpx/util/ios_flags_saver.hpp
  hpx/util/manage_config.hpp
  hpx/util/sed_transform.hpp
)

set(util_compat_headers
  hpx/traits/detail/reserve.hpp
)

set(util_sources
  manage_config.cpp
  sed_transform.cpp
)

include(HPX_AddModule)
add_hpx_module(util
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  FORCE_LINKING_GEN
  GLOBAL_HEADER_GEN ON
  EXCLUDE_FROM_GLOBAL_HEADER
    "hpx/util/detail/reserve.hpp"
  SOURCES ${util_sources}
  HEADERS ${util_headers}
  COMPAT_HEADERS ${util_compat_headers}
  DEPENDENCIES
    hpx_config
    hpx_concepts
    hpx_format
    hpx_iterator_support
  CMAKE_SUBDIRS examples tests
)
