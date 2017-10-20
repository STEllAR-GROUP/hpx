# Copyright (c) 2007-2016 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    for_each_value_proxy
   )

include_directories(${CUDA_INCLUDE_DIRS})

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  set(${test}_FLAGS
    DEPENDENCIES ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     EXCLUDE_FROM_ALL
                     HPX_PREFIX ${HPX_BUILD_PREFIX}
                     FOLDER "Tests/Regressions/Compute")

  add_hpx_regression_test("computeapi_dir" ${test} ${${test}_PARAMETERS})

  # add a custom target for this test
  add_hpx_pseudo_target(tests.regressions.computeapi_dir.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.computeapi_dir
                              tests.regressions.computeapi_dir.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.computeapi_dir.${test}
                              ${test}_test_exe)
endforeach()

