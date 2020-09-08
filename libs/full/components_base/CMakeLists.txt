# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/components_base/include
set(components_base_headers
    hpx/components_base/get_lva.hpp hpx/components_base/pinned_ptr.hpp
    hpx/components_base/traits/component_pin_support.hpp
    hpx/components_base/traits/is_component.hpp
)

# Default location is $HPX_ROOT/libs/components_base/include_compatibility
set(components_base_compat_headers
    hpx/runtime/components/pinned_ptr.hpp hpx/runtime/get_lva.hpp
    hpx/traits/component_pin_support.hpp hpx/traits/is_component.hpp
)

set(components_base_sources)

include(HPX_AddModule)
add_hpx_module(
  full components_base
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${components_base_sources}
  HEADERS ${components_base_headers}
  COMPAT_HEADERS ${components_base_compat_headers}
  DEPENDENCIES hpx_core
  CMAKE_SUBDIRS examples tests
)
