# Copyright (c) 2007-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(root "${PROJECT_SOURCE_DIR}/examples/performance_counters/sine")

###############################################################################
# add component
add_hpx_component(sine
  FOLDER "Examples/PerformanceCounters/Sine"
  HEADER_GLOB "${root}/*.h*"
  SOURCE_GLOB "${root}/sine.c*"
  AUTOGLOB
  PLUGIN)

###############################################################################
# add example executable
add_hpx_executable(sine_client
  SOURCES sine_client.cpp
  DEPENDENCIES sine_component
  FOLDER "Examples/PerformanceCounters/Sine")

# add a custom target for this example
add_hpx_pseudo_target(examples.performance_counters.sine_client)

# make pseudo-targets depend on master pseudo-target
add_hpx_pseudo_dependencies(examples.performance_counters
                            examples.performance_counters.sine_client)

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(examples.performance_counters.sine_client
                            sine_client_exe)

