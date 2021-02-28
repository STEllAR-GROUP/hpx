# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(async_headers
    hpx/async.hpp
    hpx/modules/async_distributed.hpp
    hpx/async_distributed/applier/applier.hpp
    hpx/async_distributed/applier/apply_callback.hpp
    hpx/async_distributed/applier/apply_continue_callback.hpp
    hpx/async_distributed/applier/apply_continue_fwd.hpp
    hpx/async_distributed/applier/apply_continue.hpp
    hpx/async_distributed/applier/apply.hpp
    hpx/async_distributed/applier/detail/apply_implementations_fwd.hpp
    hpx/async_distributed/applier/detail/apply_implementations.hpp
    hpx/async_distributed/applier_fwd.hpp
    hpx/async_distributed/applier/trigger.hpp
    hpx/async_distributed/apply.hpp
    hpx/async_distributed/async_callback_fwd.hpp
    hpx/async_distributed/async_callback.hpp
    hpx/async_distributed/async_continue_callback_fwd.hpp
    hpx/async_distributed/async_continue_callback.hpp
    hpx/async_distributed/async_continue_fwd.hpp
    hpx/async_distributed/async_continue.hpp
    hpx/async_distributed/async.hpp
    hpx/async_distributed/continuation_impl.hpp
    hpx/async_distributed/continuation2_impl.hpp
    hpx/async_distributed/make_continuation.hpp
    hpx/async_distributed/dataflow.hpp
    hpx/async_distributed/detail/async_implementations_fwd.hpp
    hpx/async_distributed/detail/async_implementations.hpp
    hpx/async_distributed/detail/async_unwrap_result_implementations_fwd.hpp
    hpx/async_distributed/detail/async_unwrap_result_implementations.hpp
    hpx/async_distributed/detail/sync_implementations_fwd.hpp
    hpx/async_distributed/detail/sync_implementations.hpp
    hpx/async_distributed/sync.hpp
)

# cmake-format: off
set(async_compat_headers
    hpx/apply.hpp => hpx/include/async.hpp
    hpx/dataflow.hpp => hpx/include/async.hpp
    hpx/lcos/async_callback_fwd.hpp => hpx/include/async.hpp
    hpx/lcos/async_callback.hpp => hpx/include/async.hpp
    hpx/lcos/async_continue_callback_fwd.hpp => hpx/include/async.hpp
    hpx/lcos/async_continue_callback.hpp => hpx/include/async.hpp
    hpx/lcos/async_continue_fwd.hpp => hpx/include/async.hpp
    hpx/lcos/async_continue.hpp => hpx/include/async.hpp
    hpx/lcos/async_fwd.hpp => hpx/include/async.hpp
    hpx/lcos/async.hpp => hpx/include/async.hpp
    hpx/lcos/dataflow.hpp => hpx/include/async.hpp
    hpx/lcos/sync_fwd.hpp => hpx/include/async.hpp
    hpx/lcos/sync.hpp => hpx/include/async.hpp
    hpx/runtime/actions/continuation_impl.hpp => hpx/include/async.hpp
    hpx/runtime/actions/continuation2_impl.hpp => hpx/include/async.hpp
    hpx/runtime/actions/make_continuation.hpp => hpx/include/async.hpp
    hpx/runtime/applier/applier.hpp => hpx/include/async.hpp
    hpx/runtime/applier/apply_callback.hpp => hpx/include/async.hpp
    hpx/runtime/applier/apply_continue_callback.hpp => hpx/include/async.hpp
    hpx/runtime/applier/apply_continue_fwd.hpp => hpx/include/async.hpp
    hpx/runtime/applier/apply_continue.hpp => hpx/include/async.hpp
    hpx/runtime/applier/apply_helper.hpp => hpx/include/async.hpp
    hpx/runtime/applier/apply.hpp => hpx/include/async.hpp
    hpx/runtime/applier/bind_naming_wrappers.hpp => hpx/include/async.hpp
    hpx/runtime/applier_fwd.hpp => hpx/include/async.hpp
    hpx/runtime/applier/trigger.hpp => hpx/include/async.hpp
    hpx/sync.hpp => hpx/include/async.hpp
)
# cmake-format: on

include(HPX_AddModule)
add_hpx_module(
  full async_distributed
  GLOBAL_HEADER_GEN OFF
  HEADERS ${async_headers}
  COMPAT_HEADERS ${async_compat_headers}
  DEPENDENCIES hpx_core hpx_parallelism
  MODULE_DEPENDENCIES hpx_actions hpx_naming hpx_runtime_configuration
                      hpx_runtime_local hpx_threadmanager
  CMAKE_SUBDIRS examples tests
)
