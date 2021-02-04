# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(agas_headers
    hpx/agas/addressing_service.hpp hpx/agas/agas_fwd.hpp
    hpx/agas/basename_registration.hpp hpx/agas/big_boot_barrier.hpp
    hpx/agas/state.hpp
)

# cmake-format: off
set(agas_compat_headers
    hpx/runtime/agas_fwd.hpp => hpx/agas/agas_fwd.hpp
    hpx/runtime/agas/addressing_service.hpp => hpx/agas/addressing_service.hpp
    hpx/runtime/agas/big_boot_barrier.hpp => hpx/agas/big_boot_barrier.hpp
)
# cmake-format: on

set(agas_sources addressing_service.cpp basename_registration.cpp
                 big_boot_barrier.cpp detail/interface.cpp state.cpp
)

include(HPX_AddModule)
add_hpx_module(
  full agas
  COMPATIBILITY_HEADERS OFF
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${agas_sources}
  HEADERS ${agas_headers}
  COMPAT_HEADERS ${agas_compat_headers}
  DEPENDENCIES hpx_core
  MODULE_DEPENDENCIES hpx_agas_base hpx_async_distributed hpx_naming
                      hpx_runtime_configuration hpx_runtime_local
  CMAKE_SUBDIRS examples tests
)
