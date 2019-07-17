# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

set(datastructures_headers
  hpx/datastructures/tuple.hpp
  hpx/datastructures/optional.hpp
  hpx/datastructures/detail/pack.hpp
)

set(datastructures_compat_headers
  hpx/util/detail/pack.hpp
  hpx/util/optional.hpp
  hpx/util/tuple.hpp
)

set(datastructures_sources)

include(HPX_AddModule)
add_hpx_module(datastructures
  COMPATIBILITY_HEADERS OFF
  DEPRECATION_WARNINGS
  FORCE_LINKING_GEN
  GLOBAL_HEADER_GEN ON
  EXCLUDE_FROM_GLOBAL_HEADER
    "hpx/datastructures/detail/pack.hpp"
  SOURCES ${datastructures_sources}
  HEADERS ${datastructures_headers}
  COMPAT_HEADERS ${datastructures_compat_headers}
  DEPENDENCIES hpx_config hpx_type_support
  CMAKE_SUBDIRS examples tests
)
