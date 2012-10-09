# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
# Copyright (c) 2007-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs
    agas
    actions
    lcos
    threads
    util
   )

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tests.regressions.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(tests.regressions tests.regressions.${subdir})
endforeach()

set(tests
    multiple_init
   )

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     FOLDER "Tests/Regressions/")

  add_hpx_regression_test(${test} ${${test}_PARAMETERS})  

  # add a custom target for this example
  add_hpx_pseudo_target(tests.regressions.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions
                              tests.regressions.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.${test}
                              ${test}_test_exe)
endforeach()

