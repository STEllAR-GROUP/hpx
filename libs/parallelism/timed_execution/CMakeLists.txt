# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(timed_execution_headers
    hpx/timed_execution/timed_execution.hpp
    hpx/timed_execution/timed_execution_fwd.hpp
    hpx/timed_execution/timed_executors.hpp
    hpx/timed_execution/traits/is_timed_executor.hpp
)

# cmake-format: off
set(timed_execution_compat_headers
    hpx/parallel/executors/timed_execution.hpp => hpx/include/parallel_executors.hpp
    hpx/parallel/executors/timed_execution_fwd.hpp => hpx/include/parallel_executors.hpp
    hpx/parallel/executors/timed_executors.hpp => hpx/include/parallel_executors.hpp
    hpx/traits/is_timed_executor.hpp => hpx/include/parallel_executors.hpp
)
# cmake-format: on

include(HPX_AddModule)
add_hpx_module(
  parallelism timed_execution
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  HEADERS ${timed_execution_headers}
  COMPAT_HEADERS ${timed_execution_compat_headers}
  DEPENDENCIES hpx_core
  MODULE_DEPENDENCIES hpx_execution hpx_executors hpx_threading
  CMAKE_SUBDIRS examples tests
)
