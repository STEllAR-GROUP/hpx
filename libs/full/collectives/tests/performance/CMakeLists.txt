# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs lcos osu)

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tests.performance.modules.collectives.${subdir}_perf)
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(
    tests.performance.modules.collectives
    tests.performance.modules.collectives.${subdir}_perf
  )
endforeach()
