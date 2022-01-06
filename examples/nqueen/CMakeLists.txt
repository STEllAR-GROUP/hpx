# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011 Bryce Lelbach
# Copyright (c) 2011 Vinay C Amatya
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

add_hpx_component(
  nqueen INTERNAL_FLAGS NOEXPORT
  SOURCE_GLOB "nqueen.cp*"
  HEADER_GLOB "nqueen.hp*"
  FOLDER "Examples/Applications/NQueen"
  AUTOGLOB
)

add_hpx_executable(
  nqueen_client_example INTERNAL_FLAGS
  SOURCES nqueen_client.cpp
  DEPENDENCIES nqueen_component
  FOLDER "Examples/Applications/NQueen"
)

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(examples.nqueen nqueen_client_example)

# TODO: Add non-interactive version that can be tested.
