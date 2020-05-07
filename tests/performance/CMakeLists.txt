# Copyright (c) 2007-2014 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs local)

if(HPX_WITH_NETWORKING)
  set(subdirs ${subdirs} network)
endif()

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tests.performance.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(tests.performance tests.performance.${subdir})
endforeach()
