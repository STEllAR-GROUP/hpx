# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
# Copyright (c) 2007-2017 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs build block_matrix threads util)

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  set(subdirs ${subdirs} actions agas component lcos)
endif()

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tests.regressions.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(tests.regressions tests.regressions.${subdir})
endforeach()

set(tests multiple_init multiple_init_2918 stack_size_config_4543
          unhandled_exception_582
)

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  set(tests ${tests} id_type_ref_counting_1032)
endif()

set(id_type_ref_counting_1032_PARAMETERS THREADS_PER_LOCALITY 1)
set(unhandled_exception_582_PARAMETERS THREADS_PER_LOCALITY 1)

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

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  add_hpx_regression_test(
    ""
    id_type_ref_counting_1032_4
    EXECUTABLE
    id_type_ref_counting_1032
    PSEUDO_DEPS_NAME
    id_type_ref_counting_1032
    THREADS_PER_LOCALITY
    4
  )
endif()
