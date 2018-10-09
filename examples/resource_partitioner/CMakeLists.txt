# Copyright (c) 2017 Shoshana Jakobovits
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs
  guided_pool_test
  oversubscribing_resource_partitioner
  simple_resource_partitioner
  simplest_resource_partitioner_1
  simplest_resource_partitioner_2
  async_customization
)

set(guided_pool_test_FLAGS DEPENDENCIES iostreams_component)
set(oversubscribing_resource_partitioner_FLAGS DEPENDENCIES iostreams_component)
set(simple_resource_partitioner_FLAGS DEPENDENCIES iostreams_component)

set(oversubscribing_resource_partitioner_PARAMETERS
    THREADS_PER_LOCALITY 4 "--use-pools" "--use-scheduler")
set(simple_resource_partitioner_PARAMETERS
    THREADS_PER_LOCALITY 4 "--use-pools" "--use-scheduler")
set(simplest_resource_partitioner_1_PARAMETERS
    THREADS_PER_LOCALITY 4)
set(simplest_resource_partitioner_2_PARAMETERS
    THREADS_PER_LOCALITY 4)
set(guided_pool_test_PARAMETERS
    THREADS_PER_LOCALITY 4)
set(async_customization_PARAMETERS
    THREADS_PER_LOCALITY 4)

foreach(example_program ${example_programs})
  set(sources
      ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${example_program}
                     SOURCES ${sources}
                     ${${example_program}_FLAGS}
                     FOLDER "Examples/ResourcePartitioner")

  if(MSVC)
    add_definitions(-D_USE_MATH_DEFINES)
  endif()

  # add a custom target for this example
  add_hpx_pseudo_target(examples.resource_partitioner.${example_program})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.resource_partitioner
                              examples.resource_partitioner.${example_program})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.resource_partitioner.${example_program}
                              ${example_program}_exe)

  if(HPX_WITH_TESTS AND HPX_WITH_TESTS_EXAMPLES)
    add_hpx_example_test("resource_partitioner" ${example_program} ${${example_program}_PARAMETERS})
    add_hpx_pseudo_target(tests.examples.resource_partitioner.${example_program})
    add_hpx_pseudo_dependencies(tests.examples.resource_partitioner
                                tests.examples.resource_partitioner.${example_program})
    add_hpx_pseudo_dependencies(tests.examples.resource_partitioner.${example_program}
                                ${example_program}_exe)
  endif()
endforeach()
