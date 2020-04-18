# Copyright (c) 2007-2013 Hartmut Kaiser
# Copyright (c) 2014-2015 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_AddLibrary)

################################################################################
# Decide whether to use the MPI based parcelport
################################################################################
if(HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_MPI)
  hpx_debug("add_parcelport_mpi_module")
  include(HPX_AddParcelport)
  add_parcelport(mpi
    STATIC
    SOURCES
      "${PROJECT_SOURCE_DIR}/plugins/parcelport/mpi/mpi_environment.cpp"
      "${PROJECT_SOURCE_DIR}/plugins/parcelport/mpi/parcelport_mpi.cpp"
    HEADERS
      "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/mpi/header.hpp"
      "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/mpi/locality.hpp"
      "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/mpi/mpi_environment.hpp"
      "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/mpi/receiver.hpp"
      "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/mpi/receiver_connection.hpp"
      "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/mpi/sender.hpp"
      "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/mpi/sender_connection.hpp"
      "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/mpi/tag_provider.hpp"
    DEPENDENCIES
      hpx_asio
      hpx_allocator_support
      hpx_cache
      hpx_command_line_handling
      hpx_config
      hpx_concurrency
      hpx_coroutines
      hpx_execution
      hpx_filesystem
      hpx_hardware
      hpx_memory
      hpx_performance_counters
      hpx_plugin
      hpx_program_options
      hpx_runtime_configuration
      hpx_serialization
      hpx_threading
      hpx_threadmanager
      hpx_timing
      hpx_topology
      hpx_util
      hpx_dependencies_boost
      Mpi::mpi
    INCLUDE_DIRS
      ${PROJECT_SOURCE_DIR}
    FOLDER
      "Core/Plugins/Parcelport/MPI")
elseif(HPX_MPI_WITH_FUTURES)
  hpx_debug("add_parcelport_mpi_module (reduced)")
  include(HPX_AddParcelport)
  add_parcelport(mpi
    STATIC
    SOURCES
      "${PROJECT_SOURCE_DIR}/plugins/parcelport/mpi/mpi_environment.cpp"
    HEADERS
      "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/mpi/mpi_environment.hpp"
    DEPENDENCIES
      hpx_config
      hpx_command_line_handling
      hpx_runtime_configuration
      hpx_synchronization
      hpx_util
      hpx_dependencies_boost
      Mpi::mpi
    INCLUDE_DIRS
      ${PROJECT_SOURCE_DIR}
    FOLDER
      "Core/Plugins/Parcelport/MPI")
endif()
