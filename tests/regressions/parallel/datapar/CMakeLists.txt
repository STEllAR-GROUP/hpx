# Copyright (c) 2014-2015 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    lambda_arguments_2403
   )

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
                     FOLDER "Tests/Regressions/Parallel/Datapar")

  add_hpx_regression_test("parallel" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.regressions.parallel_dir.datapar_dir.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.parallel_dir.datapar_dir
                              tests.regressions.parallel_dir.datapar_dir.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.parallel_dir.datapar_dir.${test}
                              ${test}_test_exe)
endforeach()
