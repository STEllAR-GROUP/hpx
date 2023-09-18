# Copyright (c) 2023 Chistopher Taylor
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT
   (HPX_WITH_NETWORKING
    AND HPX_WITH_PARCELPORT_OPENSHMEM
    AND HPX_WITH_PARCELPORT_OPENSHMEM_CONDUIT)
)
  return()
endif()

include(HPX_SetupOpenSHMEM)
hpx_setup_openshmem()

set(parcelport_openshmem_headers
    hpx/parcelport_openshmem/header.hpp
    hpx/parcelport_openshmem/locality.hpp
    hpx/parcelport_openshmem/receiver.hpp
    hpx/parcelport_openshmem/receiver_connection.hpp
    hpx/parcelport_openshmem/sender.hpp
    hpx/parcelport_openshmem/sender_connection.hpp
    hpx/parcelport_openshmem/tag_provider.hpp
)

set(parcelport_openshmem_sources locality.cpp parcelport_openshmem.cpp)

include(HPX_AddModule)
add_hpx_module(
  full parcelport_openshmem
  GLOBAL_HEADER_GEN ON
  SOURCES ${parcelport_openshmem_sources}
  HEADERS ${parcelport_openshmem_headers}
  DEPENDENCIES hpx_core hpx_openshmem_base PkgConfig::openshmem
               ${openshmem_additional_dependencies}
  MODULE_DEPENDENCIES hpx_actions hpx_command_line_handling hpx_parcelset
  CMAKE_SUBDIRS examples tests
)

set(HPX_STATIC_PARCELPORT_PLUGINS
    ${HPX_STATIC_PARCELPORT_PLUGINS} parcelport_openshmem
    CACHE INTERNAL "" FORCE
)
