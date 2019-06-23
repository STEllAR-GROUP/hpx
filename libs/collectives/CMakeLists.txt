# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/collectives/include
set(collectives_headers
  hpx/collectives/all_to_all.hpp
)

# Default location is $HPX_ROOT/libs/collectives/src
set(collectives_sources
  collectives.cpp
)

include(HPX_AddModule)
add_hpx_module(collectives
    SOURCES ${collectives_sources}
    HEADERS ${collectives_headers}
    DEPENDENCIES hpx_config hpx_preprocessor
    CMAKE_SUBDIRS examples tests
)
