# Copyright (c) 2007-2015 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2013 Jeroen Habraken
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(components io memory papi)

foreach(component ${components})
  add_hpx_pseudo_target(components.performance_counters.${component})
  add_subdirectory(${component})
  add_hpx_pseudo_dependencies(
    components.performance_counters
    components.performance_counters.${component}
  )
endforeach()
