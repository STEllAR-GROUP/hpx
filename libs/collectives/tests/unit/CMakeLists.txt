# Copyright (c) 2019 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    all_reduce
    all_to_all
   )

set(all_reduce_PARAMETERS LOCALITIES 2)
set(all_to_all_PARAMETERS LOCALITIES 2)

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
                     FOLDER "Tests/Modules/Unit/Collectives")

  add_hpx_test("tests.unit.modules.collectives" ${test} ${${test}_PARAMETERS})

  # add a custom target for this test
  add_hpx_pseudo_target(tests.unit.modules.collectives.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.modules.collectives
                              tests.unit.modules.collectives.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.modules.collectives.${test}
                              ${test}_test)
endforeach()


