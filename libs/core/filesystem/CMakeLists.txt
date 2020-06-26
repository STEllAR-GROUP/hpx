# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(__boost_filesystem)
if(HPX_FILESYSTEM_WITH_BOOST_FILESYSTEM_COMPATIBILITY)
  set(__boost_filesystem Boost::filesystem)
endif()

set(filesystem_headers hpx/modules/filesystem.hpp)

set(filesystem_compat_headers hpx/filesystem.hpp)

include(HPX_AddModule)
add_hpx_module(
  core filesystem
  GLOBAL_HEADER_GEN OFF
  COMPATIBILITY_HEADERS ON # Added in 1.5.0
  DEPRECATION_WARNINGS
  HEADERS ${filesystem_headers}
  COMPAT_HEADERS ${filesystem_compat_headers}
  MODULE_DEPENDENCIES hpx_config
  DEPENDENCIES ${__boost_filesystem}
  CMAKE_SUBDIRS examples tests
)
