# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/concurrency/include
set(concurrency_headers
    hpx/concurrency/barrier.hpp
    hpx/concurrency/cache_line_data.hpp
    hpx/concurrency/concurrentqueue.hpp
    hpx/concurrency/deque.hpp
    hpx/concurrency/detail/contiguous_index_queue.hpp
    hpx/concurrency/detail/freelist.hpp
    hpx/concurrency/detail/tagged_ptr_pair.hpp
    hpx/concurrency/spinlock.hpp
    hpx/concurrency/spinlock_pool.hpp
)

# Default location is $HPX_ROOT/libs/concurrency/include_compatibility
# cmake-format: off
set(concurrency_compat_headers
    hpx/compat/barrier.hpp => hpx/modules/concurrency.hpp
    hpx/concurrency.hpp => hpx/modules/concurrency.hpp
    hpx/util/lockfree/concurrentqueue.hpp => hpx/modules/concurrency.hpp
    hpx/util/lockfree/deque.hpp => hpx/modules/concurrency.hpp
    hpx/util/lockfree/freelist.hpp => hpx/modules/concurrency.hpp
    hpx/util/spinlock.hpp => hpx/modules/concurrency.hpp
    hpx/util/spinlock_pool.hpp => hpx/modules/concurrency.hpp
)
# cmake-format: on

# Default location is $HPX_ROOT/libs/concurrency/src
set(concurrency_sources barrier.cpp)

include(HPX_AddModule)
add_hpx_module(
  core concurrency
  COMPATIBILITY_HEADERS ON # Added in 1.5.0
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${concurrency_sources}
  HEADERS ${concurrency_headers}
  COMPAT_HEADERS ${concurrency_compat_headers}
  MODULE_DEPENDENCIES
    hpx_assertion
    hpx_concepts
    hpx_config
    hpx_datastructures
    hpx_execution_base
    hpx_errors
    hpx_hashing
    hpx_itt_notify
    hpx_thread_support
  CMAKE_SUBDIRS examples tests
)
