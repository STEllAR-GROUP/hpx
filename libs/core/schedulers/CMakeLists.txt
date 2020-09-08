# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(schedulers_headers
    hpx/schedulers/deadlock_detection.hpp
    hpx/schedulers/local_priority_queue_scheduler.hpp
    hpx/schedulers/local_queue_scheduler.hpp
    hpx/schedulers/lockfree_queue_backends.hpp
    hpx/schedulers/maintain_queue_wait_times.hpp
    hpx/schedulers/queue_helpers.hpp
    hpx/schedulers/queue_holder_numa.hpp
    hpx/schedulers/queue_holder_thread.hpp
    hpx/schedulers/shared_priority_queue_scheduler.hpp
    hpx/schedulers/static_priority_queue_scheduler.hpp
    hpx/schedulers/static_queue_scheduler.hpp
    hpx/schedulers/thread_queue.hpp
    hpx/schedulers/thread_queue_mc.hpp
    hpx/modules/schedulers.hpp
)

set(schedulers_compat_headers
    hpx/runtime/threads/policies/deadlock_detection.hpp
    hpx/runtime/threads/policies/local_priority_queue_scheduler.hpp
    hpx/runtime/threads/policies/local_queue_scheduler.hpp
    hpx/runtime/threads/policies/lockfree_queue_backends.hpp
    hpx/runtime/threads/policies/maintain_queue_wait_times.hpp
    hpx/runtime/threads/policies/queue_helpers.hpp
    hpx/runtime/threads/policies/shared_priority_queue_scheduler.hpp
    hpx/runtime/threads/policies/static_priority_queue_scheduler.hpp
    hpx/runtime/threads/policies/static_queue_scheduler.hpp
    hpx/runtime/threads/policies/thread_queue.hpp
    hpx/runtime/threads/policies/thread_queue_mc.hpp
    hpx/runtime/threads/policies/schedulers.hpp
)

set(schedulers_sources deadlock_detection.cpp maintain_queue_wait_times.cpp)

include(HPX_AddModule)
add_hpx_module(
  core schedulers
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN OFF
  SOURCES ${schedulers_sources}
  HEADERS ${schedulers_headers}
  COMPAT_HEADERS ${schedulers_compat_headers}
  SOURCES ${schedulers_sources}
  MODULE_DEPENDENCIES
    hpx_config
    hpx_assertion
    hpx_errors
    hpx_format
    hpx_functional
    hpx_logging
    hpx_threading_base
  CMAKE_SUBDIRS examples tests
)
