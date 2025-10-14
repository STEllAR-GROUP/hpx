# Copyright (c) 2019-2025 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(tag_invoke_headers
    hpx/functional/detail/invoke.hpp
    hpx/functional/detail/tag_fallback_invoke.hpp
    hpx/functional/detail/tag_priority_invoke.hpp
    hpx/functional/invoke_result.hpp
    hpx/functional/macros.hpp
    hpx/functional/tag_invoke.hpp
    hpx/functional/traits/is_invocable.hpp
)
set(tag_invoke_macro_headers hpx/functional/macros.hpp)

# cmake-format: off
set(tag_invoke_compat_headers
    hpx/functional/result_of.hpp => hpx/functional/invoke_result.hpp
    hpx/functional/traits/is_callable.hpp => hpx/functional/traits/is_invocable.hpp
)
# cmake-format: on

set(tag_invoke_sources)

include(HPX_AddModule)
add_hpx_module(
  core tag_invoke
  GLOBAL_HEADER_GEN ON
  GLOBAL_HEADER_MODULE_GEN ON
  SOURCES ${tag_invoke_sources}
  HEADERS ${tag_invoke_headers}
  MACRO_HEADERS ${tag_invoke_macro_headers}
  COMPAT_HEADERS ${tag_invoke_compat_headers}
  ADD_TO_GLOBAL_HEADER "hpx/functional/detail/tag_fallback_invoke.hpp"
                       "hpx/functional/detail/tag_priority_invoke.hpp"
  MODULE_DEPENDENCIES hpx_config hpx_type_support
  CMAKE_SUBDIRS examples tests
)
