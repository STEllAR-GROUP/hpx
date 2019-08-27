# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

set(filesystem_headers "hpx/filesystem.hpp")

include(HPX_AddModule)
add_hpx_module(filesystem
  FORCE_LINKING_GEN
  GLOBAL_HEADER_GEN OFF
  HEADERS ${filesystem_headers}
  DEPENDENCIES hpx_config
  CMAKE_SUBDIRS examples tests
)
