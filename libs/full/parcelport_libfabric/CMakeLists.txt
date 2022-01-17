# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT (HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_LIBFABRIC))
  return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Configuring options
include(HPX_SetupLibfabric)

set(libfabric_libraries Libfabric::libfabric)
if(HPX_PARCELPORT_LIBFABRIC_WITH_BOOTSTRAPPING)
  set(libfabric_libraries ${libfabric_libraries} Pmi::pmi)
endif()

set(parcelport_libfabric_headers
    hpx/parcelport_libfabric/connection_handler.hpp
    hpx/parcelport_libfabric/fabric_error.hpp
    hpx/parcelport_libfabric/header.hpp
    hpx/parcelport_libfabric/libfabric_controller.hpp
    hpx/parcelport_libfabric/libfabric_region_provider.hpp
    hpx/parcelport_libfabric/locality.hpp
    hpx/parcelport_libfabric/parcelport_libfabric.hpp
    hpx/parcelport_libfabric/parcelport_logging.hpp
    hpx/parcelport_libfabric/performance_counter.hpp
    hpx/parcelport_libfabric/pinned_memory_vector.hpp
    hpx/parcelport_libfabric/rdma_locks.hpp
    hpx/parcelport_libfabric/readers_writers_mutex.hpp
    hpx/parcelport_libfabric/receiver.hpp
    hpx/parcelport_libfabric/rma_base.hpp
    hpx/parcelport_libfabric/rma_memory_pool.hpp
    hpx/parcelport_libfabric/rma_memory_region.hpp
    hpx/parcelport_libfabric/rma_memory_region_traits.hpp
    hpx/parcelport_libfabric/rma_receiver.hpp
    hpx/parcelport_libfabric/sender.hpp
    hpx/parcelport_libfabric/unordered_map.hpp
)

# cmake-format: off
set(parcelport_libfabric_compat_headers)
# cmake-format: on

set(parcelport_libfabric_sources parcelport_libfabric.cpp receiver.cpp
                                 rma_receiver.cpp sender.cpp
)

include(HPX_AddModule)
add_hpx_module(
  full parcelport_libfabric
  GLOBAL_HEADER_GEN ON
  SOURCES ${parcelport_libfabric_sources}
  HEADERS ${parcelport_libfabric_headers}
  COMPAT_HEADERS ${parcelport_libfabric_compat_headers}
  DEPENDENCIES hpx_core ${libfabric_libraries}
  MODULE_DEPENDENCIES hpx_actions hpx_command_line_handling hpx_parcelset
  CMAKE_SUBDIRS examples tests
)

set(HPX_STATIC_PARCELPORT_PLUGINS
    ${HPX_STATIC_PARCELPORT_PLUGINS} parcelport_libfabric
    CACHE INTERNAL "" FORCE
)
