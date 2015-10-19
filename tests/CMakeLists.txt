# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs "")

if(HPX_WITH_TESTS_BENCHMARKS)
  set(subdirs ${subdirs} performance)
  set(performance_FLAGS ON)
endif()

if(HPX_WITH_TESTS_REGRESSIONS)
  set(subdirs ${subdirs} regressions)
  set(regressions_FLAGS ON)
endif()

if(HPX_WITH_TESTS_UNIT)
  set(subdirs ${subdirs} unit)
  set(unit_FLAGS ON)
endif()

if(HPX_WITH_TESTS_HEADERS)
  set(subdirs ${subdirs} headers)
endif()

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tests.${subdir})
  add_subdirectory(${subdir})
  if(${subdir}_FLAGS)
    add_hpx_pseudo_dependencies(tests tests.${subdir})
  endif()
endforeach()

