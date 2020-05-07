# Copyright (c) 2012 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

add_subdirectory(jacobi_component)

set(jacobi_sources jacobi.cpp)

set(jacobi_FLAGS COMPONENT_DEPENDENCIES iostreams jacobi)

# add jacobi executable
add_hpx_executable(
  jacobi_simple INTERNAL_FLAGS MODULE jacobi
  SOURCES ${jacobi_sources} ${jacobi_FLAGS}
  FOLDER "Examples/Jacobi"
)

add_hpx_example_target_dependencies("jacobi" jacobi_simple)

# TODO: Fix example. Not added to unit tests until fixed.
