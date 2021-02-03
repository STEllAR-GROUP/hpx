# Copyright (c) 2017 Shoshana Jakobovits
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_EXAMPLES)
  add_hpx_pseudo_target(examples.modules.resource_partitioner)
  add_hpx_pseudo_dependencies(
    examples.modules examples.modules.resource_partitioner
  )
  if(HPX_WITH_TESTS
     AND HPX_WITH_TESTS_EXAMPLES
     AND HPX_RESOURCE_PARTITIONER_WITH_TESTS
  )
    add_hpx_pseudo_target(tests.examples.modules.resource_partitioner)
    add_hpx_pseudo_dependencies(
      tests.examples.modules tests.examples.modules.resource_partitioner
    )
  endif()
else()
  return()
endif()

set(example_programs simplest_resource_partitioner_1
                     simplest_resource_partitioner_2
)

if(HPX_WITH_DISTRIBUTED_RUNTIME
   AND HPX_WITH_MAX_CPU_COUNT
   AND HPX_WITH_SHARED_PRIORITY_SCHEDULER
)
  set(example_programs
      ${example_programs} async_customization
      oversubscribing_resource_partitioner simple_resource_partitioner
  )

  # Intel compilers have trouble compiling this example
  if(NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    set(example_programs ${example_programs} guided_pool_test)
  endif()
endif()

set(guided_pool_test_FLAGS DEPENDENCIES iostreams_component)
set(oversubscribing_resource_partitioner_FLAGS DEPENDENCIES iostreams_component)
set(simple_resource_partitioner_FLAGS DEPENDENCIES iostreams_component)

set(oversubscribing_resource_partitioner_PARAMETERS
    THREADS_PER_LOCALITY 4 "--use-pools" "--pool-threads" 1
)
set(simple_resource_partitioner_PARAMETERS THREADS_PER_LOCALITY 4 "--use-pools"
                                           "--pool-threads" 1
)
set(simplest_resource_partitioner_1_PARAMETERS THREADS_PER_LOCALITY 4)
set(simplest_resource_partitioner_2_PARAMETERS THREADS_PER_LOCALITY 4)
set(guided_pool_test_PARAMETERS THREADS_PER_LOCALITY 4)
set(async_customization_PARAMETERS THREADS_PER_LOCALITY 4)

foreach(example_program ${example_programs})
  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${example_program} INTERNAL_FLAGS
    SOURCES ${sources} ${${example_program}_FLAGS}
    FOLDER "Examples/ResourcePartitioner"
  )

  if(MSVC)
    add_definitions(-D_USE_MATH_DEFINES)
  endif()

  add_hpx_example_target_dependencies(
    "modules.resource_partitioner" ${example_program}
  )

  if(HPX_WITH_TESTS
     AND HPX_WITH_TESTS_EXAMPLES
     AND HPX_RESOURCE_PARTITIONER_WITH_TESTS
  )
    add_hpx_example_test(
      "modules.resource_partitioner" ${example_program}
      ${${example_program}_PARAMETERS}
    )
  endif()
endforeach()
