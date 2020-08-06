# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(type_support_headers
    hpx/type_support/detail/wrap_int.hpp
    hpx/type_support/always_void.hpp
    hpx/type_support/decay.hpp
    hpx/type_support/detected.hpp
    hpx/type_support/equality.hpp
    hpx/type_support/identity.hpp
    hpx/type_support/lazy_conditional.hpp
    hpx/type_support/lazy_enable_if.hpp
    hpx/type_support/pack.hpp
    hpx/type_support/static.hpp
    hpx/type_support/unwrap_ref.hpp
    hpx/type_support/unused.hpp
    hpx/type_support/void_guard.hpp
)

set(type_support_compat_headers
    hpx/type_support.hpp
    hpx/util/always_void.hpp
    hpx/util/decay.hpp
    hpx/util/detected.hpp
    hpx/util/identity.hpp
    hpx/util/lazy_conditional.hpp
    hpx/util/lazy_enable_if.hpp
    hpx/util/static.hpp
    hpx/util/unwrap_ref.hpp
    hpx/util/unused.hpp
    hpx/util/void_guard.hpp
)

set(type_support_sources)

include(HPX_AddModule)
add_hpx_module(
  type_support
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${type_support_sources}
  HEADERS ${type_support_headers}
  COMPAT_HEADERS ${type_support_compat_headers}
  DEPENDENCIES hpx_assertion hpx_config
  CMAKE_SUBDIRS examples tests
)
