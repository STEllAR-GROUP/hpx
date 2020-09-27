# Copyright (c) 2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(pack_traversal_headers
    hpx/pack_traversal/detail/container_category.hpp
    hpx/pack_traversal/detail/pack_traversal_impl.hpp
    hpx/pack_traversal/detail/pack_traversal_async_impl.hpp
    hpx/pack_traversal/detail/unwrap_impl.hpp
    hpx/pack_traversal/pack_traversal.hpp
    hpx/pack_traversal/pack_traversal_async.hpp
    hpx/pack_traversal/traits/pack_traversal_rebind_container.hpp
    hpx/pack_traversal/unwrap.hpp
)

# cmake-format: off
set(pack_traversal_compat_headers
    hpx/traits/pack_traversal_rebind_container.hpp => hpx/modules/pack_traversal.hpp
    hpx/util/pack_traversal.hpp => hpx/modules/pack_traversal.hpp
    hpx/util/pack_traversal_async.hpp => hpx/modules/pack_traversal.hpp
    hpx/util/unwrap.hpp => hpx/modules/pack_traversal.hpp
)
# cmake-format: on

include(HPX_AddModule)
add_hpx_module(
  parallelism pack_traversal
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  HEADERS ${pack_traversal_headers}
  COMPAT_HEADERS ${pack_traversal_compat_headers}
  DEPENDENCIES hpx_core
  CMAKE_SUBDIRS examples tests
)
