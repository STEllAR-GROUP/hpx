# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/util/include
set(util_headers
  hpx/util/await_traits.hpp
  hpx/util/manage_config.hpp
  hpx/util/memory_chunk.hpp
  hpx/util/safe_lexical_cast.hpp
)

# Default location is $HPX_ROOT/libs/util/src
set(util_sources
  manage_config.cpp
)

include(HPX_AddModule)
add_hpx_module(util
    COMPATIBILITY_HEADERS OFF
    DEPRECATION_WARNINGS
    FORCE_LINKING_GEN
    GLOBAL_HEADER_GEN ON
    SOURCES ${util_sources}
    HEADERS ${util_headers}
    COMPAT_HEADERS ${util_compat_headers}
    DEPENDENCIES
      hpx_assertion
      hpx_config
      hpx_concepts
      hpx_iterator_support
      hpx_type_support
    CMAKE_SUBDIRS examples tests
)
