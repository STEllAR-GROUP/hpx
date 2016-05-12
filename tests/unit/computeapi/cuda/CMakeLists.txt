# Copyright (c) 2016 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
   )

if(HPX_WITH_CUDA)
  set(tests ${tests}
      default_executor
      triad_for_each
     )
  set(default_executor_CUDA On)
  set(triad_for_each_CUDA On)
endif()

foreach(test ${tests})
  if(${${test}_CUDA})
    set(sources
      ${test}.cu)
  else()
    set(sources
        ${test}.cpp)
  endif()

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     EXCLUDE_FROM_ALL
                     HPX_PREFIX ${HPX_BUILD_PREFIX}
                     FOLDER "Tests/Unit/Compute/CUDA")

  add_hpx_unit_test("cuda_" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.unit.computeapi.cuda_.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.computeapi.cuda_
                              tests.unit.computeapi.cuda_.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.computeapi.cuda_.${test}
                              ${test}_test_exe)
endforeach()

