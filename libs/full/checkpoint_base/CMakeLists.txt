# Copyright (c) 2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(checkpoint_base_headers hpx/checkpoint_base/checkpoint_data.hpp)

set(checkpoint_base_sources checkpoint_data.cpp)

include(HPX_AddModule)
add_hpx_module(
  full checkpoint_base
  GLOBAL_HEADER_GEN ON
  SOURCES ${checkpoint_base_sources}
  HEADERS ${checkpoint_base_headers}
  DEPENDENCIES hpx_core
  CMAKE_SUBDIRS tests
)
