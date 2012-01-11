# Copyright (c) 2007-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs
    lcos
   )

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tests.units.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(tests.units tests.units.${subdir})
endforeach()

