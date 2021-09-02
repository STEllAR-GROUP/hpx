# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
# Copyright (c) 2007-2012 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs build lcos util)

foreach(subdir ${subdirs})
  if(NOT TARGET tests.unit.${subdir})
    hpx_local_add_pseudo_target(tests.unit.${subdir})
  endif()
  add_subdirectory(${subdir})
  hpx_local_add_pseudo_dependencies(tests.unit tests.unit.${subdir})
endforeach()
