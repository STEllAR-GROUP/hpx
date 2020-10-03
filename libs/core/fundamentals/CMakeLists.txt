# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/fundamentals/include
set(fundamentals_headers
    hpx/concepts/concepts.hpp
    hpx/config/lambda_capture.hpp
    hpx/config/non_copyable.hpp
    hpx/functional/detail/invoke.hpp
    hpx/iterator_support/range.hpp
    hpx/type_support/detail/wrap_int.hpp
    hpx/type_support/always_void.hpp
    hpx/type_support/decay.hpp
    hpx/type_support/lazy_conditional.hpp
    hpx/type_support/lazy_enable_if.hpp
    hpx/type_support/pack.hpp
    hpx/type_support/unwrap_ref.hpp
    hpx/type_support/void_guard.hpp
)

# Default location is $HPX_ROOT/libs/fundamentals/include_compatibility
set(fundamentals_compat_headers
    hpx/concepts.hpp => hpx/concepts/concepts.hpp
    hpx/traits/concepts.hpp => hpx/concepts/concepts.hpp
    hpx/util/always_void.hpp => hpx/type_support/always_void.hpp
    hpx/util/decay.hpp => hpx/type_support/decay.hpp
    hpx/util/lazy_conditional.hpp => hpx/type_support/lazy_conditional.hpp
    hpx/util/lazy_enable_if.hpp => hpx/type_support/lazy_enable_if.hpp
    hpx/util/range.hpp => hpx/iterator_support/range.hpp
    hpx/util/unwrap_ref.hpp => hpx/type_support/unwrap_ref.hpp
    hpx/util/void_guard.hpp => hpx/type_support/void_guard.hpp
)

# Default location is $HPX_ROOT/libs/fundamentals/src
set(fundamentals_sources)

include(HPX_AddModule)
add_hpx_module(
  core fundamentals
  COMPATIBILITY_HEADERS ON # Added in 1.5.0
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${fundamentals_sources}
  HEADERS ${fundamentals_headers}
  COMPAT_HEADERS ${fundamentals_compat_headers}
  MODULE_DEPENDENCIES hpx_config
  CMAKE_SUBDIRS examples tests
)
