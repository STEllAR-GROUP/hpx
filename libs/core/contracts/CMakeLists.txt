# Copyright (c) 2025 The STE||AR-Group
# Copyright (c) 2025 Alexandros Papadakis
#
# Copyright (c) 2025 Panagiotis Syskakis
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/contracts/include
set(contracts_headers
    hpx/contracts.hpp hpx/modules/contracts.hpp
)

# Default location is $HPX_ROOT/libs/contracts/include_compatibility
# cmake-format: off
set(contracts_compat_headers
    # Add any legacy header mappings here if needed in the future
    # hpx/old_contracts.hpp => hpx/contracts.hpp
)
# cmake-format: on

# Default location is $HPX_ROOT/libs/contracts/src
set(contracts_sources
    # Add source files here when implementation is added
    # contracts.cpp
)

include(HPX_AddModule)
add_hpx_module(
  core contracts
  GLOBAL_HEADER_GEN OFF
  SOURCES ${contracts_sources}
  HEADERS ${contracts_headers}
  COMPAT_HEADERS ${contracts_compat_headers}
  MODULE_DEPENDENCIES hpx_config hpx_preprocessor hpx_assertion
  CMAKE_SUBDIRS examples tests
)