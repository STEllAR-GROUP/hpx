# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include_directories(${HDF5_INCLUDE_DIR} ${HDF5_CPP_INCLUDE_DIR})

# add a custom target for this example
add_hpx_pseudo_target(examples.sheneos.sheneos)

add_subdirectory(sheneos)

# make pseudo-targets depend on master pseudo-target
add_hpx_pseudo_dependencies(examples.sheneos examples.sheneos.sheneos)

###############################################################################
source_group("Source Files" FILES sheneos_client.cpp sheneos_test.cpp)

# add example executable
add_hpx_executable(sheneos_client
  SOURCES sheneos_client.cpp
  DEPENDENCIES sheneos_component
  FOLDER "Examples/ShenEOS")

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(examples.sheneos sheneos_client_exe)

add_hpx_executable(sheneos_test
  SOURCES sheneos_test.cpp
  DEPENDENCIES sheneos_component
  FOLDER "Examples/ShenEOS")

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(examples.sheneos sheneos_test_exe)

