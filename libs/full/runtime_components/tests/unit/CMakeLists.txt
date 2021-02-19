# Copyright (c) 2007-2021 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# add subdirectories
set(subdirs agas components)

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tests.unit.modules.runtime_components.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(
    tests.unit.modules.runtime_components
    tests.unit.modules.runtime_components.${subdir}
  )
endforeach()
