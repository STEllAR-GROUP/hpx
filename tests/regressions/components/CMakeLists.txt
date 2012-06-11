# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_subdirectory(components)

set(tests)

# Windows requires all symbols to be resolved at link time, this test does
# not make sense on this platform.
if(NOT MSVC)
  set(tests
      undefined_symbol_at_runtime
     )
endif()

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     FOLDER "Tests/Regressions/Components")

  add_hpx_regression_test(${test} ${${test}_PARAMETERS})  

  # add a custom target for this example
  add_hpx_pseudo_target(tests.regressions.components.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.components
                              tests.regressions.components.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.components.${test}
                              ${test}_test_exe)
endforeach()

