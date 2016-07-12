# Copyright (c) 2016 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    iterator_facade
    stencil3_iterator
    transform_iterator
    transform_iterator2
    zip_iterator
   )

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     EXCLUDE_FROM_ALL
                     HPX_PREFIX ${HPX_BUILD_PREFIX}
                     FOLDER "Tests/Unit/Util/Iterator")

  add_hpx_unit_test("util.iterator" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.unit.util.iteratordir.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.util.iteratordir
                              tests.unit.util.iteratordir.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.util.iteratordir.${test}
                              ${test}_test_exe)
endforeach()
