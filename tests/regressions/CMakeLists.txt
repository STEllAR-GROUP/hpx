# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
# Copyright (c) 2007-2017 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs threads util)

foreach(subdir ${subdirs})
  hpx_local_add_pseudo_target(tests.regressions.${subdir})
  add_subdirectory(${subdir})
  hpx_local_add_pseudo_dependencies(
    tests.regressions tests.regressions.${subdir}
  )
endforeach()

set(tests multiple_init_local multiple_init_local_2918
          stack_size_config_local_4543 unhandled_exception_local_582
)

set(unhandled_exception_local_582_PARAMETERS THREADS_PER_LOCALITY 1)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  hpx_local_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Regressions/"
  )

  hpx_local_add_regression_test("" ${test} ${${test}_PARAMETERS})

endforeach()
