# Copyright (c) 2007-2015 Hartmut Kaiser
# Copyright (c) 2013,2016 Maciej Brodowicz
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    discover_counters_1787
    uptime_1737)

if(HPX_WITH_PAPI)
  set(tests ${tests}
    papi_counters_active_interface
    papi_counters_basic_functions
    papi_counters_segfault_1890)
endif()

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  if(${test} MATCHES "^papi_.*")
    add_hpx_executable(${test}_test
                       SOURCES ${sources}
                       COMPONENT_DEPENDENCIES "papi_counters"
                       ${${test}_FLAGS}
                       EXCLUDE_FROM_ALL
                       HPX_PREFIX ${HPX_BUILD_PREFIX}
                       FOLDER "Tests/Regressions/PerformanceCounters")
  else()
    add_hpx_executable(${test}_test
                       SOURCES ${sources}
                       ${${test}_FLAGS}
                       EXCLUDE_FROM_ALL
                       HPX_PREFIX ${HPX_BUILD_PREFIX}
                       FOLDER "Tests/Regressions/PerformanceCounters")
  endif()

  add_hpx_regression_test("performance_counters" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.regressions.performance_counters_dir.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.performance_counters_dir
                              tests.regressions.performance_counters_dir.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.performance_counters_dir.${test}
                              ${test}_test_exe)
endforeach()


