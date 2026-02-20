# Copyright (c) 2026 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_TESTS_BENCHMARKS)
  add_hpx_executable(
    concurrent_structure_performance
    SOURCES concurrent_structure_performance.cpp
    HPX_PREFIX ${HPX_BUILD_PREFIX}
  )
endif()
