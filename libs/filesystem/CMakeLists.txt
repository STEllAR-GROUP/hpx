# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

set(__filesystem_compatibility_default ON)
if(HPX_WITH_CXX17_FILESYSTEM)
  set(__filesystem_compatibility_default OFF)
endif()

# Compatibility with using Boost.FileSystem, introduced in V1.4.0
hpx_option(HPX_FILESYSTEM_WITH_BOOST_FILESYSTEM_COMPATIBILITY
  BOOL "Enable Boost.FileSystem compatibility. (default: ${__filesystem_compatibility_default})"
  ON ADVANCED CATEGORY "Modules")

set(__boost_filesystem)
if(HPX_FILESYSTEM_WITH_BOOST_FILESYSTEM_COMPATIBILITY)

  hpx_add_config_define_namespace(
    DEFINE HPX_FILESYSTEM_HAVE_BOOST_FILESYSTEM_COMPATIBILITY
    NAMESPACE FILESYSTEM)

  find_package(Boost ${Boost_MINIMUM_VERSION}
    QUIET MODULE
    COMPONENTS filesystem)
  if(Boost_FILESYSTEM_FOUND)
    set(__boost_filesystem ${Boost_LIBRARIES})
  else()
    hpx_error("Could not find Boost.Filesystem. Provide a boost installation "
      "including the filesystem library or use a compiler with support for the "
      "C++17 filesystem library")
  endif()
  hpx_libraries(${Boost_LIBRARIES})
else()
  if(NOT HPX_WITH_CXX17_FILESYSTEM)
    hpx_error("Could not find std::filesystem. Use a compiler with support for "
      "the C++17 filesystem library or enable Boost.FileSystem compatibility "
      "(set HPX_FILESYSTEM_WITH_BOOST_FILESYSTEM_COMPATIBILITY to ON)")
  endif()
endif()

set(filesystem_headers "hpx/filesystem.hpp")

include(HPX_AddModule)
add_hpx_module(filesystem
  FORCE_LINKING_GEN
  GLOBAL_HEADER_GEN OFF
  HEADERS ${filesystem_headers}
  DEPENDENCIES
    hpx_config
    ${__boost_filesystem}
  CMAKE_SUBDIRS examples tests
)

if(HPX_FILESYSTEM_WITH_BOOST_FILESYSTEM_COMPATIBILITY)
  hpx_info("    Boost library found: filesystem")
endif()
