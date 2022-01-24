# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
# Copyright (c) 2007-2017 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs build block_matrix threads util)

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  set(subdirs ${subdirs} component lcos)
endif()

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tests.regressions.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(tests.regressions tests.regressions.${subdir})
endforeach()

set(tests stack_size_config_4543)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Regressions/"
  )

  add_hpx_regression_test("" ${test} ${${test}_PARAMETERS})

endforeach()
