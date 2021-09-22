# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(async_local_headers
    hpx/async_local/apply.hpp hpx/async_local/async.hpp
    hpx/async_local/async_fwd.hpp hpx/async_local/dataflow.hpp
    hpx/async_local/sync.hpp hpx/async_local/sync_fwd.hpp
)

include(HPX_AddModule)
add_hpx_module(
  core async_local
  GLOBAL_HEADER_GEN ON
  HEADERS ${async_local_headers}
  MODULE_DEPENDENCIES hpx_async_base hpx_execution hpx_executors hpx_config
  CMAKE_SUBDIRS examples tests
)
