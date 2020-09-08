# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(coroutines_headers
    hpx/coroutines/coroutine.hpp
    hpx/coroutines/coroutine_fwd.hpp
    hpx/coroutines/stackless_coroutine.hpp
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
    hpx/coroutines/detail/coroutine_stackful_self.hpp
    hpx/coroutines/detail/coroutine_stackless_self.hpp
    hpx/coroutines/detail/get_stack_pointer.hpp
    hpx/coroutines/detail/posix_utility.hpp
    hpx/coroutines/detail/swap_context.hpp
    hpx/coroutines/detail/tss.hpp
    hpx/coroutines/thread_enums.hpp
    hpx/coroutines/thread_id_type.hpp
)

set(coroutines_compat_headers
    hpx/coroutines.hpp hpx/runtime/threads/coroutines/coroutine.hpp
    hpx/runtime/threads/coroutines/coroutine_fwd.hpp
    hpx/runtime/threads/thread_enums.hpp hpx/runtime/threads/thread_id_type.hpp
)

set(coroutines_sources
    detail/context_base.cpp
    detail/coroutine_impl.cpp
    detail/coroutine_self.cpp
    detail/posix_utility.cpp
    detail/tss.cpp
    swapcontext.cpp
    thread_enums.cpp
)

if(MSVC)
  if(HPX_WITH_SWAP_CONTEXT_EMULATION)
    enable_language(ASM)
    set(switch_to_fiber_source
        "${CMAKE_CURRENT_SOURCE_DIR}/src/switch_to_fiber.asm"
    )
    set(switch_to_fiber_object
        "${CMAKE_CURRENT_BINARY_DIR}/switch_to_fiber.obj"
    )
    add_custom_command(
      OUTPUT "${switch_to_fiber_object}"
      COMMAND "${CMAKE_ASM_MASM_COMPILER}" /Fo "${switch_to_fiber_object}"
              /nologo /c "${switch_to_fiber_source}"
      DEPENDS "${switch_to_fiber_source}"
      VERBATIM
    )
  endif()
endif()

include(HPX_AddModule)
add_hpx_module(
  core coroutines
  COMPATIBILITY_HEADERS ON # Added in 1.5.0
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${coroutines_sources}
  HEADERS ${coroutines_headers} OBJECTS "${switch_to_fiber_object}"
  COMPAT_HEADERS ${coroutines_compat_headers}
  MODULE_DEPENDENCIES hpx_assertion hpx_config hpx_errors hpx_format
                      hpx_functional hpx_util
  CMAKE_SUBDIRS examples tests
)
