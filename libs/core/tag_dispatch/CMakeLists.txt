# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(tag_dispatch_headers
    hpx/functional/detail/invoke.hpp
    hpx/functional/invoke_result.hpp
    hpx/functional/tag_fallback_dispatch.hpp
    hpx/functional/tag_dispatch.hpp
    hpx/functional/tag_priority_dispatch.hpp
    hpx/functional/traits/is_invocable.hpp
)

# cmake-format: off
set(tag_dispatch_compat_headers
    hpx/functional/result_of.hpp => hpx/functional/invoke_result.hpp
    hpx/functional/traits/is_callable.hpp => hpx/functional/traits/is_invocable.hpp
)
# cmake-format: on

set(tag_dispatch_sources)

include(HPX_AddModule)
add_hpx_module(
  core tag_dispatch
  GLOBAL_HEADER_GEN ON
  SOURCES ${tag_dispatch_sources}
  HEADERS ${tag_dispatch_headers}
  COMPAT_HEADERS ${tag_dispatch_compat_headers}
  MODULE_DEPENDENCIES hpx_config hpx_type_support
  CMAKE_SUBDIRS examples tests
)
