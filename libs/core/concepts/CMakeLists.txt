# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(concepts_headers hpx/concepts/concepts.hpp hpx/concepts/has_member_xxx.hpp
                     hpx/concepts/has_xxx.hpp
)

set(concepts_compat_headers
    hpx/concepts.hpp hpx/traits/concepts.hpp hpx/traits/has_member_xxx.hpp
    hpx/traits/has_xxx.hpp
)

set(concepts_sources)

include(HPX_AddModule)
add_hpx_module(
  core concepts
  COMPATIBILITY_HEADERS ON # Added in 1.5.0
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${concepts_sources}
  HEADERS ${concepts_headers}
  COMPAT_HEADERS ${concepts_compat_headers}
  MODULE_DEPENDENCIES hpx_config hpx_preprocessor hpx_type_support
  CMAKE_SUBDIRS examples tests
)
