# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs "")

if(HPX_BUILD_TESTS_BENCHMARKS)
  set(subdirs
    ${subdirs} performance)
endif()

if(HPX_BUILD_TESTS_REGRESSIONS)
  set(subdirs
    ${subdirs} regressions)
endif()

if(HPX_BUILD_TESTS_UNIT)
  set(subdirs
    ${subdirs} unit)
endif()

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tests.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(tests tests.${subdir})
endforeach()

