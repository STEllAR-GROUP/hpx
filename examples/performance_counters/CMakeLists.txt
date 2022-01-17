# Copyright (c) 2007-2017 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

add_subdirectory(sine)

set(example_programs access_counter_set simplest_performance_counter)

set(access_counter_set_PARAMETERS THREADS_PER_LOCALITY 4)
set(simplest_performance_counter_PARAMETERS THREADS_PER_LOCALITY 4)

set(access_counter_set_FLAGS COMPONENT_DEPENDENCIES iostreams)

foreach(example_program ${example_programs})
  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  set(${example_program}_FLAGS DEPENDENCIES iostreams_component)

  # add example executable
  add_hpx_executable(
    ${example_program} INTERNAL_FLAGS
    SOURCES ${sources} ${${example_program}_FLAGS}
    FOLDER "Examples/PerformanceCounters/${example_program}"
  )

  add_hpx_example_target_dependencies("performance_counters" ${example_program})

  if(HPX_WITH_TESTS AND HPX_WITH_TESTS_EXAMPLES)
    add_hpx_example_test(
      "performance_counters" ${example_program}
      ${${example_program}_PARAMETERS}
    )
  endif()

endforeach()
