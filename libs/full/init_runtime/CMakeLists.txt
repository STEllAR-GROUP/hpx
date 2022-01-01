# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(init_runtime_headers
    hpx/init.hpp
    hpx/hpx_finalize.hpp
    hpx/hpx_init.hpp
    hpx/hpx_init_impl.hpp
    hpx/hpx_init_params.hpp
    hpx/hpx_main_winsocket.hpp
    hpx/hpx_start.hpp
    hpx/hpx_start_impl.hpp
    hpx/hpx_suspend.hpp
    hpx/hpx_user_main_config.hpp
    hpx/init_runtime/detail/init_logging.hpp
    hpx/init_runtime/detail/run_or_start.hpp
    hpx/init_runtime/pre_main.hpp
)

set(init_runtime_sources hpx_init.cpp hpx_main_winsocket.cpp init_logging.cpp
                         pre_main.cpp
)

include(HPX_AddModule)
add_hpx_module(
  full init_runtime
  GLOBAL_HEADER_GEN OFF
  HEADERS ${init_runtime_headers}
  SOURCES ${init_runtime_sources}
  DEPENDENCIES HPX::hpx_local
  MODULE_DEPENDENCIES
    hpx_async_distributed
    hpx_collectives
    hpx_command_line_handling
    hpx_config
    hpx_naming
    hpx_performance_counters
    hpx_runtime_distributed
  CMAKE_SUBDIRS examples tests
)
