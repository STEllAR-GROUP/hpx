# Copyright (c) 2007-2013 Hartmut Kaiser
# Copyright (c) 2014-2015 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_PARCELPORT_TCP)
  hpx_debug("add_parcelport_tcp_module")
  include(HPX_AddParcelport)
  add_parcelport(
    tcp STATIC
    SOURCES
      "${PROJECT_SOURCE_DIR}/plugins/parcelport/tcp/connection_handler_tcp.cpp"
      "${PROJECT_SOURCE_DIR}/plugins/parcelport/tcp/parcelport_tcp.cpp"
    HEADERS
      "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/tcp/connection_handler.hpp"
      "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/tcp/locality.hpp"
      "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/tcp/receiver.hpp"
      "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/tcp/sender.hpp"
    DEPENDENCIES
      hpx_actions
      hpx_allocator_support
      hpx_asio
      hpx_assertion
      hpx_cache
      hpx_concurrency
      hpx_config
      hpx_coroutines
      hpx_errors
      hpx_execution
      hpx_functional
      hpx_hardware
      hpx_memory
      hpx_performance_counters
      hpx_plugin
      hpx_program_options
      hpx_runtime_local
      hpx_serialization
      hpx_threading
      hpx_threadmanager
      hpx_timing
      hpx_topology
      hpx_util
    INCLUDE_DIRS "${PROJECT_SOURCE_DIR}"
    FOLDER "Core/Plugins/Parcelport/Tcp"
  )
endif()
