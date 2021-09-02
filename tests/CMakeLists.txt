# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs "")

if(HPXLocal_WITH_TESTS_BENCHMARKS)
  set(subdirs ${subdirs} performance)
  set(performance_FLAGS ON)
endif()

if(HPXLocal_WITH_TESTS_REGRESSIONS)
  set(subdirs ${subdirs} regressions)
  set(regressions_FLAGS ON)
endif()

if(HPXLocal_WITH_TESTS_UNIT)
  set(subdirs ${subdirs} unit)
  set(unit_FLAGS ON)
endif()

foreach(subdir ${subdirs})
  add_subdirectory(${subdir})
endforeach()
