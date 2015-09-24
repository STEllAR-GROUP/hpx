# Copyright (c) 2013 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    mem_fn_derived_test
    mem_fn_dm_test
#    mem_fn_eq_test
    mem_fn_rv_test
    mem_fn_test
    mem_fn_unary_addr_test
    mem_fn_void_test
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
                     FOLDER "Tests/Unit/Util/Mem_Fn")

  add_hpx_unit_test("util.mem_fn" ${test} ${${test}_PARAMETERS})

  set_property(TARGET ${test}_test_exe APPEND
               PROPERTY COMPILE_DEFINITIONS
               "HPX_LIMIT=8")

  # add a custom target for this example
  add_hpx_pseudo_target(tests.unit.util.mem_fndir.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.util.mem_fndir
                              tests.unit.util.mem_fndir.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.util.mem_fndir.${test}
                              ${test}_test_exe)
endforeach()
