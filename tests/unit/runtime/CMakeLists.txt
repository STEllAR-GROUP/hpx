# Copyright (c) 2018 Nikunj Gupta
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT MSVC)

  set(tests
      runtime_initialization)

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
                       FOLDER "Tests/Unit/Runtime/")

    add_hpx_unit_test("runtime" ${test} ${${test}_PARAMETERS})

    # add a custom target for this example
    add_hpx_pseudo_target(tests.unit.runtime.${test})

    # make pseudo-targets depend on master pseudo-target
    add_hpx_pseudo_dependencies(tests.unit.runtime
                                tests.unit.runtime.${test})

    # add dependencies to pseudo-target
    add_hpx_pseudo_dependencies(tests.unit.runtime.${test}
                                ${test}_test_exe)
  endforeach()

endif()
