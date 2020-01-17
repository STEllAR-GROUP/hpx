# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(coroutines_headers
  hpx/coroutines/coroutine.hpp
  hpx/coroutines/coroutine_fwd.hpp
  hpx/coroutines/detail/combined_tagged_state.hpp
  hpx/coroutines/detail/context_base.hpp
  hpx/coroutines/detail/context_generic_context.hpp
  hpx/coroutines/detail/context_impl.hpp
  hpx/coroutines/detail/context_linux_x86.hpp
  hpx/coroutines/detail/context_posix.hpp
  hpx/coroutines/detail/context_windows_fibers.hpp
  hpx/coroutines/detail/coroutine_accessor.hpp
  hpx/coroutines/detail/coroutine_impl.hpp
  hpx/coroutines/detail/coroutine_self.hpp
  hpx/coroutines/detail/get_stack_pointer.hpp
  hpx/coroutines/detail/posix_utility.hpp
  hpx/coroutines/detail/swap_context.hpp
  hpx/coroutines/detail/tss.hpp
  hpx/coroutines/thread_enums.hpp
  hpx/coroutines/thread_id_type.hpp
  )

set(coroutines_compat_headers
  hpx/runtime/threads/coroutines/coroutine.hpp
  hpx/runtime/threads/coroutines/coroutine_fwd.hpp
  hpx/runtime/threads/coroutines/detail/context_base.hpp
  hpx/runtime/threads/coroutines/detail/context_generic_context.hpp
  hpx/runtime/threads/coroutines/detail/context_impl.hpp
  hpx/runtime/threads/coroutines/detail/context_linux_x86.hpp
  hpx/runtime/threads/coroutines/detail/context_posix.hpp
  hpx/runtime/threads/coroutines/detail/context_windows_fibers.hpp
  hpx/runtime/threads/coroutines/detail/coroutine_accessor.hpp
  hpx/runtime/threads/coroutines/detail/coroutine_impl.hpp
  hpx/runtime/threads/coroutines/detail/coroutine_self.hpp
  hpx/runtime/threads/coroutines/detail/get_stack_pointer.hpp
  hpx/runtime/threads/coroutines/detail/posix_utility.hpp
  hpx/runtime/threads/coroutines/detail/swap_context.hpp
  hpx/runtime/threads/coroutines/detail/tss.hpp
  hpx/runtime/threads/detail/combined_tagged_state.hpp
  hpx/runtime/threads/thread_enums.hpp
  hpx/runtime/threads/thread_id_type.hpp
  )

set(coroutines_sources
  detail/context_base.cpp
  detail/coroutine_impl.cpp
  detail/tss.cpp
  swapcontext.cpp
  )

include(HPX_AddModule)
add_hpx_module(coroutines
  COMPATIBILITY_HEADERS ON  # Added in 1.4.0
  DEPRECATION_WARNINGS
  FORCE_LINKING_GEN
  GLOBAL_HEADER_GEN ON
  EXCLUDE_FROM_GLOBAL_HEADER
    "hpx/threading/coroutines/detail/context_linux_x86.hpp"
    "hpx/threading/coroutines/detail/context_posix.hpp"
    "hpx/threading/coroutines/detail/context_windows_fibers.hpp"
  SOURCES ${coroutines_sources}
  HEADERS ${coroutines_headers}
  COMPAT_HEADERS ${coroutines_compat_headers}
  DEPENDENCIES
    hpx_assertion
    hpx_basic_execution
    hpx_config
    hpx_format
    hpx_functional
    hpx_util
  CMAKE_SUBDIRS examples tests
)
