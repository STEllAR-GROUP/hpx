# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/collectives/include
set(collectives_headers
  hpx/collectives/all_reduce.hpp
  hpx/collectives/all_to_all.hpp
)

# Default location is $HPX_ROOT/libs/collectives/src
set(collectives_sources
)

include(HPX_AddModule)
add_hpx_module(collectives
    DEPRECATION_WARNINGS
    FORCE_LINKING_GEN
    SOURCES ${collectives_sources}
    HEADERS ${collectives_headers}
    DEPENDENCIES
      hpx_assertion
      hpx_config
      hpx_errors
      hpx_format
      hpx_logging
      hpx_preprocessor
      hpx_timing
      hpx_type_support
      hpx_util
    CMAKE_SUBDIRS examples tests
)
