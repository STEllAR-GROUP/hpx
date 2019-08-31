# Copyright (c) 2014-2018 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    bulk_then_execute_3182
    handled_exception_2959
    is_executor_1691
   )

if(HPX_WITH_DATAPAR)
  set(tests ${tests}
      lambda_arguments_2403
      lambda_return_type_2402
     )
endif()

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
                     FOLDER "Tests/Regressions/Modules/Executors")

  add_hpx_regression_test("modules.executors" ${test} ${${test}_PARAMETERS})

endforeach()
