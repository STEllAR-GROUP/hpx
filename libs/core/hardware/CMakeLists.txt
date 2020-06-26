# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/hardware/include
set(hardware_headers
    hpx/hardware/cpuid/linux_x86.hpp
    hpx/hardware/cpuid/msvc.hpp
    hpx/hardware/timestamp/bgq.hpp
    hpx/hardware/timestamp/linux_generic.hpp
    hpx/hardware/timestamp/linux_x86_32.hpp
    hpx/hardware/timestamp/linux_x86_64.hpp
    hpx/hardware/timestamp/msvc.hpp
    hpx/hardware/bit_manipulation.hpp
    hpx/hardware/cpuid.hpp
    hpx/hardware/timestamp.hpp
)

# Default location is $HPX_ROOT/libs/hardware/include_compatibility
set(hardware_compat_headers
    hpx/util/hardware/cpuid/linux_x86.hpp
    hpx/util/hardware/cpuid/msvc.hpp
    hpx/util/hardware/timestamp/bgq.hpp
    hpx/util/hardware/timestamp/linux_generic.hpp
    hpx/util/hardware/timestamp/linux_x86_32.hpp
    hpx/util/hardware/timestamp/linux_x86_64.hpp
    hpx/util/hardware/timestamp/msvc.hpp
    hpx/util/hardware/bit_manipulation.hpp
    hpx/util/hardware/cpuid.hpp
    hpx/util/hardware/timestamp.hpp
)

# Default location is $HPX_ROOT/libs/hardware/src
set(hardware_sources)

include(HPX_AddModule)

add_hpx_module(
  core hardware
  COMPATIBILITY_HEADERS ON # Added in 1.5.0
  DEPRECATION_WARNINGS
  EXCLUDE_FROM_GLOBAL_HEADER
    "hpx/hardware/cpuid/linux_x86.hpp"
    "hpx/hardware/cpuid/msvc.hpp"
    "hpx/hardware/timestamp/bgq.hpp"
    "hpx/hardware/timestamp/linux_generic.hpp"
    "hpx/hardware/timestamp/linux_x86_32.hpp"
    "hpx/hardware/timestamp/linux_x86_64.hpp"
    "hpx/hardware/timestamp/msvc.hpp"
  SOURCES ${hardware_sources}
  HEADERS ${hardware_headers}
  COMPAT_HEADERS ${hardware_compat_headers}
  MODULE_DEPENDENCIES hpx_config
  CMAKE_SUBDIRS examples tests
)
