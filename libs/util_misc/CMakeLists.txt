# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

set(util_misc_headers
  hpx/util_misc/detail/reserve.hpp
  hpx/util_misc/calculate_fanout.hpp
  hpx/util_misc/get_and_reset_value.hpp
  hpx/util_misc/insert_checked.hpp
)

set(util_misc_compat_headers
  hpx/traits/detail/reserve.hpp
  hpx/util/calculate_fanout.hpp
  hpx/util/get_and_reset_value.hpp
  hpx/util/insert_checked.hpp
)

set(util_misc_sources)

include(HPX_AddModule)
add_hpx_module(util_misc
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  FORCE_LINKING_GEN
  GLOBAL_HEADER_GEN ON
  EXCLUDE_FROM_GLOBAL_HEADER
    "hpx/util_misc/detail/reserve.hpp"
  SOURCES ${util_misc_sources}
  HEADERS ${util_misc_headers}
  COMPAT_HEADERS ${util_misc_compat_headers}
  DEPENDENCIES hpx_config hpx_concepts hpx_iterator_support
  CMAKE_SUBDIRS examples tests
)
