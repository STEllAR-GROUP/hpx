# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/serializable_function/include
set(serializable_function_headers
    hpx/serializable_function/serialization/detail/serializable_basic_function.hpp
    hpx/serializable_function/serialization/detail/vtable/serializable_function_vtable.hpp
    hpx/serializable_function/serialization/detail/vtable/serializable_vtable.hpp
    hpx/serializable_function/serialization/serializable_function.hpp
    hpx/serializable_function/serialization/serializable_unique_function.hpp
)

# Default location is $HPX_ROOT/libs/serializable_function/include_compatibility
# cmake-format: off
set(serializable_function_compat_headers)
# cmake-format: on

# Default location is $HPX_ROOT/libs/serializable_function/src
set(serializable_function_sources)

include(HPX_AddModule)
add_hpx_module(
  core serializable_function
  GLOBAL_HEADER_GEN ON
  EXCLUDE_FROM_GLOBAL_HEADER
  SOURCES ${serializable_function_sources}
  HEADERS ${serializable_function_headers}
  COMPAT_HEADERS ${serializable_function_compat_headers}
  MODULE_DEPENDENCIES
    hpx_assertion
    hpx_config
    hpx_datastructures
    hpx_debugging
    hpx_errors
    hpx_functional
    hpx_itt_notify
    hpx_serialization
    hpx_type_support
    # CMAKE_SUBDIRS examples tests
)
