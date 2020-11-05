# Copyright (c) 2007-2017 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_EXAMPLES_HDF5)
  if(NOT HDF5_CXX_LIBRARIES)
    hpx_error("The ShenEos example requires HDF C++ library")
  endif()

  # add a custom target for this example
  add_hpx_pseudo_target(examples.sheneos.sheneos_)

  add_subdirectory(sheneos)

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.sheneos examples.sheneos.sheneos_)

  # ############################################################################
  # add example executable
  add_hpx_executable(
    sheneos_client INTERNAL_FLAGS
    SOURCES sheneos_client.cpp
    DEPENDENCIES sheneos_component
    FOLDER "Examples/ShenEOS"
  )

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.sheneos sheneos_client)

  add_hpx_executable(
    sheneos_test INTERNAL_FLAGS
    SOURCES sheneos_test.cpp
    DEPENDENCIES sheneos_component
    FOLDER "Examples/ShenEOS"
  )

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.sheneos sheneos_test)

  # TODO: Fix example. Not added to unit tests until fixed.
endif()
