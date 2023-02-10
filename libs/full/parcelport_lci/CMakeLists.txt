# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT (HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_LCI))
  return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(parcelport_lci_headers
    hpx/parcelport_lci/header.hpp hpx/parcelport_lci/locality.hpp
    hpx/parcelport_lci/receiver.hpp hpx/parcelport_lci/sender.hpp
    hpx/parcelport_lci/sender_connection.hpp
)

# cmake-format: off
set(parcelport_lci_compat_headers)
# cmake-format: on

set(parcelport_lci_sources locality.cpp parcelport_lci.cpp)

include(HPX_SetupLCI)
hpx_setup_lci()

include(HPX_AddModule)
add_hpx_module(
  full parcelport_lci
  GLOBAL_HEADER_GEN ON
  SOURCES ${parcelport_lci_sources}
  HEADERS ${parcelport_lci_headers}
  COMPAT_HEADERS ${parcelport_lci_compat_headers}
  DEPENDENCIES hpx_core hpx_dependencies_boost LCI::LCI
  MODULE_DEPENDENCIES hpx_actions hpx_command_line_handling hpx_parcelset
  CMAKE_SUBDIRS examples tests
)

set(HPX_STATIC_PARCELPORT_PLUGINS
    ${HPX_STATIC_PARCELPORT_PLUGINS} parcelport_lci
    CACHE INTERNAL "" FORCE
)
