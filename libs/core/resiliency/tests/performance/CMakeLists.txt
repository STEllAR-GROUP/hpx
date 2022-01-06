# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# add subdirectories
set(subdirs replay replicate)

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tests.performance.modules.resiliency.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(
    tests.performance.modules.resiliency
    tests.performance.modules.resiliency.${subdir}
  )
endforeach()
