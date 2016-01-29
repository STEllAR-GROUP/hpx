# Copyright (c) 2007-2014 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs
    algorithms
    osu
   )

if(HPX_WITH_CXX11_LAMBDAS)
  set(subdirs
    ${subdirs}
    network_storage
   )
endif()

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tests.performance.network.${subdir}_perf)
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(tests.performance.network tests.performance.network.${subdir}_perf)
endforeach()

