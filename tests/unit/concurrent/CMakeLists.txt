# Copyright (c) 2016 John Biddiscombe
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# add tests
set(tests
    unordered_set
    unordered_map
    concurrent_vector
)

set(unordered_set_FLAGS DEPENDENCIES iostreams_component)
set(unordered_map_FLAGS DEPENDENCIES iostreams_component)
set(concurrent_vector_FLAGS DEPENDENCIES iostreams_component)

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  set(${test}_PARAMETERS THREADS_PER_LOCALITY 4)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(concurrent_${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     EXCLUDE_FROM_ALL
                     HPX_PREFIX ${HPX_BUILD_PREFIX}
                     FOLDER "Tests/Unit/Concurrent/")

  add_hpx_unit_test("concurrent" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.unit.concurrent.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.concurrent
                              tests.unit.concurrent.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.concurrent.${test}
                              concurrent_${test}_test_exe)
endforeach()
