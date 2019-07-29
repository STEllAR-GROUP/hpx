# Copyright (c) 2007-2015 Hartmut Kaiser
# Copyright (c) 2013,2016 Maciej Brodowicz
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    discover_counters_1787
    statistics_2666
    uptime_1737)

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

endforeach()


