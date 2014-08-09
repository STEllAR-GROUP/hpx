# Copyright (c) 2014 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    all_of
    any_of
    copy
    copyif
    copyn
    count
    countif
    equal
    equal_binary
    fill
    filln
    find
    findend
    findfirstof
    findif
    findifnot
    foreach
    foreachn
    generate
    generaten
    mismatch
    mismatch_binary
    move
    none_of
    reduce_
    reverse
    reverse_copy
    rotate
    rotate_copy
    swapranges
    task_region
    transform
    transform_binary
    transform_reduce
   )

set(task_region_FLAGS DEPENDENCIES iostreams_component)

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  set(${test}_PARAMETERS THREADS_PER_LOCALITY 4)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
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
