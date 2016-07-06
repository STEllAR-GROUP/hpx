# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
# Copyright (c) 2007-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs
    actions
    agas
    build
    component
    concurrent
    diagnostics
    lcos
    parcelset
    performance_counter
    serialization
    threads
    traits
    util
   )

if(HPX_WITH_SECURITY)
  set(subdirs ${subdirs} security)
endif()

if(HPX_WITH_APEX)
  set(subdirs ${subdirs} apex)
endif()

if(HPX_WITH_CXX11_LAMBDAS)
  set(subdirs ${subdirs} parallel)
endif()

set(subdirs ${subdirs} computeapi)

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tests.unit.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(tests.unit tests.unit.${subdir})
endforeach()

