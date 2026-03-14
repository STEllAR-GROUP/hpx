# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(prefix_headers hpx/prefix/find_prefix.hpp hpx/prefix/macros.hpp)

set(prefix_macro_headers hpx/prefix/macros.hpp)

# cmake-format: off
set(prefix_compat_headers
    hpx/util/find_prefix.hpp => hpx/modules/prefix.hpp
)
# cmake-format: on

set(prefix_sources find_prefix.cpp)

include(HPX_AddModule)
add_hpx_module(
  core prefix
  GLOBAL_HEADER_GEN ON
  GLOBAL_HEADER_MODULE_GEN ON
  SOURCES ${prefix_sources}
  HEADERS ${prefix_headers}
  MACRO_HEADERS ${prefix_macro_headers}
  COMPAT_HEADERS ${prefix_compat_headers}
  MODULE_DEPENDENCIES
    hpx_assertion
    hpx_config
    hpx_errors
    hpx_filesystem
    hpx_plugin
    hpx_preprocessor
    hpx_string_util
    hpx_type_support
  CMAKE_SUBDIRS tests
)
