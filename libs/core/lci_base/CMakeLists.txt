# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT (HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_LCI))
  return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# setup LCI, if necessary
include(HPX_SetupLCI)
hpx_setup_lci()

# Default location is $HPX_ROOT/libs/lci_base/include
set(lci_base_headers hpx/lci_base/lci_environment.hpp)

set(lci_base_sources lci_environment.cpp)

include(HPX_AddModule)
add_hpx_module(
  core lci_base
  GLOBAL_HEADER_GEN ON
  SOURCES ${lci_base_sources}
  HEADERS ${lci_base_headers}
  MODULE_DEPENDENCIES hpx_logging hpx_runtime_configuration hpx_string_util
                      hpx_synchronization hpx_util
  DEPENDENCIES LCI::LCI LCI::LCT
  CMAKE_SUBDIRS examples tests
)
