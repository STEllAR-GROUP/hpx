# Copyright (c) 2019-2025 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Default location is $HPX_ROOT/libs/memory/include
set(memory_headers
    hpx/memory/detail/sp_convertible.hpp hpx/memory/intrusive_ptr.hpp
    hpx/memory/shared_array.hpp hpx/memory/serialization/intrusive_ptr.hpp
)

# Default location is $HPX_ROOT/libs/memory/src
set(memory_sources)

# cmake-format: off
set(memory_compat_headers
    hpx/runtime/serialization/intrusive_ptr.hpp => hpx/memory/serialization/intrusive_ptr.hpp
)
# cmake-format: on

include(HPX_AddModule)
add_hpx_module(
  core memory
  GLOBAL_HEADER_GEN ON
  GLOBAL_HEADER_MODULE_GEN ON
  SOURCES ${memory_sources}
  HEADERS ${memory_headers}
  COMPAT_HEADERS ${memory_compat_headers}
  MODULE_DEPENDENCIES hpx_config hpx_assertion hpx_serialization
  CMAKE_SUBDIRS examples tests
)
