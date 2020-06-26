# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(string_util_headers
    hpx/string_util/case_conv.hpp hpx/string_util/classification.hpp
    hpx/string_util/split.hpp hpx/string_util/trim.hpp
)

include(HPX_AddModule)
add_hpx_module(
  core string_util
  GLOBAL_HEADER_GEN ON
  HEADERS ${string_util_headers}
  CMAKE_SUBDIRS examples tests
)
