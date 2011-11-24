# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include_directories(${HDF5_CPP_INCLUDE_DIR})

add_subdirectory(sheneos)

source_group("Source Files" FILES sheneos_test.cpp)

add_hpx_executable(sheneos_test
  SOURCES sheneos_test.cpp
  DEPENDENCIES sheneos_component)

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(examples.sheneos sheneos_test_exe)

