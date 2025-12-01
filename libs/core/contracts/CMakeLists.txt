# Copyright (c) 2025 The STE||AR-Group
# Copyright (c) 2025 Alexandros Papadakis
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(HPX_WITH_CXX26_CONTRACTS)
  # Map existing HPX_ASSERT to contract_assert
  hpx_option(
    HPX_CONTRACTS_WITH_ASSERTS_AS_CONTRACT_ASSERTS BOOL
    "Map existing HPX_ASERT to contract_assert. (default: ON)" ON ADVANCED
    CATEGORY "Modules"
    MODULE CONTRACTS
  )

  if(HPX_CONTRACTS_WITH_ASSERTS_AS_CONTRACT_ASSERTS)
    hpx_add_config_define_namespace(
      DEFINE HPX_CONTRACTS_HAVE_ASSERTS_AS_CONTRACT_ASSERTS NAMESPACE CONTRACTS
    )
  endif()
endif()

# Default location is $HPX_ROOT/libs/contracts/include
set(contracts_headers hpx/contracts.hpp hpx/contracts/macros.hpp)
set(contracts_macro_headers hpx/contracts/macros.hpp)

# Default location is $HPX_ROOT/libs/contracts/src
set(contracts_sources)

include(HPX_AddModule)
add_hpx_module(
  core contracts
  GLOBAL_HEADER_GEN ON
  GLOBAL_HEADER_MODULE_GEN ON
  SOURCES ${contracts_sources}
  HEADERS ${contracts_headers}
  MACRO_HEADERS ${contracts_macro_headers}
  COMPAT_HEADERS ${contracts_compat_headers}
  MODULE_DEPENDENCIES hpx_config hpx_preprocessor hpx_assertion
  CMAKE_SUBDIRS examples tests
)
