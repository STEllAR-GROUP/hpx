# Copyright (c) 2009-2010 Dylan Stark
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2007-2012 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

# ##############################################################################
set(heartbeat_HEADERS)

set(heartbeat_SOURCES heartbeat.cpp)

if(MSVC)
  set(heartbeat_HEADERS ${heartbeat_HEADERS} win_perf_counters.hpp
                        hpx_counters.hpp
  )

  set(heartbeat_SOURCES ${heartbeat_SOURCES} win_perf_counters.cpp)
endif()

add_hpx_executable(
  heartbeat INTERNAL_FLAGS
  SOURCES ${heartbeat_SOURCES}
  HEADERS ${heartbeat_HEADERS}
  FOLDER "Examples/HeartBeat"
  DEPENDENCIES iostreams_component
)

# add pseudo-target dependencies
add_hpx_pseudo_dependencies(examples.heartbeat heartbeat)

# ##############################################################################
add_hpx_executable(
  heartbeat_console INTERNAL_FLAGS
  SOURCES heartbeat_console.cpp
  FOLDER "Examples/HeartBeat"
  DEPENDENCIES iostreams_component
)

# add pseudo-target dependencies
add_hpx_pseudo_dependencies(examples.heartbeat heartbeat_console)

# TODO: Add non-interactive version that can be tested.
