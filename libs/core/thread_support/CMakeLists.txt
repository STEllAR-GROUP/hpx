# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(thread_support_headers
    hpx/thread_support/set_thread_name.hpp
    hpx/thread_support/thread_specific_ptr.hpp
    hpx/thread_support/assert_owns_lock.hpp
    hpx/thread_support/unlock_guard.hpp
    hpx/thread_support/atomic_count.hpp
)

set(thread_support_compat_headers
    hpx/thread_support.hpp hpx/util/assert_owns_lock.hpp
    hpx/util/atomic_count.hpp hpx/util/set_thread_name.hpp
    hpx/util/thread_specific_ptr.hpp hpx/util/unlock_guard.hpp
)

set(thread_support_sources set_thread_name.cpp)

include(HPX_AddModule)
add_hpx_module(
  core thread_support
  COMPATIBILITY_HEADERS ON # Added in 1.5.0
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${thread_support_sources}
  HEADERS ${thread_support_headers}
  COMPAT_HEADERS ${thread_support_compat_headers}
  MODULE_DEPENDENCIES hpx_assertion hpx_config hpx_concepts hpx_type_support
  CMAKE_SUBDIRS examples tests
)
