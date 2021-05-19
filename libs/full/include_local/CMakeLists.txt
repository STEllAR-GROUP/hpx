# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(include_local_headers
    hpx/local/algorithm.hpp
    hpx/local/any.hpp
    hpx/local/chrono.hpp
    hpx/local/execution.hpp
    hpx/local/functional.hpp
    hpx/local/future.hpp
    hpx/local/memory.hpp
    hpx/local/numeric.hpp
    hpx/local/optional.hpp
    hpx/local/runtime.hpp
    hpx/local/task_block.hpp
    hpx/local/tuple.hpp
    hpx/local/type_traits.hpp
)

include(HPX_AddModule)
add_hpx_module(
  full include_local
  GLOBAL_HEADER_GEN OFF
  HEADERS ${include_local_headers}
  DEPENDENCIES hpx_core hpx_parallelism
  MODULE_DEPENDENCIES hpx_runtime_local hpx_thread_executors
  CMAKE_SUBDIRS examples tests
)
