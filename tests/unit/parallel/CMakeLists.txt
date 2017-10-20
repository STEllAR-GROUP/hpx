# Copyright (c) 2007-2015 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# add tests
set(tests
    spmd_block
    task_block
    task_block_executor
    task_block_par
   )

set(task_block_FLAGS DEPENDENCIES iostreams_component)
set(task_block_executor_FLAGS DEPENDENCIES iostreams_component)
set(task_block_par_FLAGS DEPENDENCIES iostreams_component)

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  set(${test}_PARAMETERS THREADS_PER_LOCALITY 4)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     EXCLUDE_FROM_ALL
                     HPX_PREFIX ${HPX_BUILD_PREFIX}
                     FOLDER "Tests/Unit/Parallel/")

  add_hpx_unit_test("parallel" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.unit.parallel.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.parallel
                              tests.unit.parallel.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.parallel.${test}
                              ${test}_test_exe)
endforeach()

# add subdirectories
set(subdirs
    algorithms
    container_algorithms
    datapar_algorithms
    executors
    segmented_algorithms
   )

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tests.unit.parallel.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(tests.unit.parallel tests.unit.parallel.${subdir})
endforeach()
