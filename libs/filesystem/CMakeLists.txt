# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

set(__boost_filesystem)
if(NOT HPX_WITH_CXX17_FILESYSTEM)
  find_package(Boost ${Boost_MINIMUM_VERSION}
    QUIET MODULE
    COMPONENTS filesystem)
  if(Boost_FILESYSTEM_FOUND)
    set(__boost_filesystem ${Boost_LIBRARIES})
  else()
    hpx_error("Could not find Boost.Filesystem. Provide a boost installation including the filesystem library or use a compiler with support for the C++17 filesystem library")
  endif()
  hpx_libraries(${Boost_LIBRARIES})
endif()

set(filesystem_headers "hpx/filesystem.hpp")

include(HPX_AddModule)
add_hpx_module(filesystem
  FORCE_LINKING_GEN
  GLOBAL_HEADER_GEN OFF
  HEADERS ${filesystem_headers}
  DEPENDENCIES hpx_config ${__boost_filesystem}
  CMAKE_SUBDIRS examples tests
)
