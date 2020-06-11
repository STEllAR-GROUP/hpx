# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/functional/include
set(functional_headers
    hpx/functional/deferred_call.hpp
    hpx/functional/detail/basic_function.hpp
    hpx/functional/detail/empty_function.hpp
    hpx/functional/detail/function_registration.hpp
    hpx/functional/detail/invoke.hpp
    hpx/functional/detail/reset_function.hpp
    hpx/functional/detail/vtable/callable_vtable.hpp
    hpx/functional/detail/vtable/copyable_vtable.hpp
    hpx/functional/detail/vtable/function_vtable.hpp
    hpx/functional/detail/vtable/vtable.hpp
    hpx/functional/bind.hpp
    hpx/functional/bind_back.hpp
    hpx/functional/bind_front.hpp
    hpx/functional/first_argument.hpp
    hpx/functional/function.hpp
    hpx/functional/function_ref.hpp
    hpx/functional/invoke.hpp
    hpx/functional/invoke_fused.hpp
    hpx/functional/invoke_result.hpp
    hpx/functional/mem_fn.hpp
    hpx/functional/one_shot.hpp
    hpx/functional/protect.hpp
    hpx/functional/tag_invoke.hpp
    hpx/functional/unique_function.hpp
    hpx/functional/serialization/detail/serializable_basic_function.hpp
    hpx/functional/serialization/detail/vtable/serializable_function_vtable.hpp
    hpx/functional/serialization/detail/vtable/serializable_vtable.hpp
    hpx/functional/serialization/serializable_function.hpp
    hpx/functional/serialization/serializable_unique_function.hpp
    hpx/functional/traits/get_action_name.hpp
    hpx/functional/traits/get_function_address.hpp
    hpx/functional/traits/get_function_annotation.hpp
    hpx/functional/traits/is_action.hpp
    hpx/functional/traits/is_bind_expression.hpp
    hpx/functional/traits/is_invocable.hpp
    hpx/functional/traits/is_placeholder.hpp
)

# Default location is $HPX_ROOT/libs/functional/include_compatibility
set(functional_compat_headers
    hpx/functional/result_of.hpp
    hpx/functional/traits/is_callable.hpp
    hpx/util/bind.hpp
    hpx/util/bind_back.hpp
    hpx/util/bind_front.hpp
    hpx/util/deferred_call.hpp
    hpx/util/first_argument.hpp
    hpx/util/function.hpp
    hpx/util/function_ref.hpp
    hpx/util/invoke.hpp
    hpx/util/invoke_fused.hpp
    hpx/util/mem_fn.hpp
    hpx/util/one_shot.hpp
    hpx/util/protect.hpp
    hpx/util/result_of.hpp
    hpx/util/serializable_function.hpp
    hpx/util/serializable_unique_function.hpp
    hpx/util/unique_function.hpp
)

# Default location is $HPX_ROOT/libs/functional/src
set(functional_sources basic_function.cpp empty_function.cpp)

include(HPX_AddModule)
add_hpx_module(
  core functional
  COMPATIBILITY_HEADERS ON # Added in 1.5.0
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN
    ON
    EXCLUDE
    "hpx/functional/traits/get_function_address.hpp"
    "hpx/functional/traits/get_function_annotation.hpp"
    "hpx/functional/traits/is_action.hpp"
    "hpx/functional/traits/is_bind_expression.hpp"
    "hpx/functional/traits/is_callable.hpp"
    "hpx/functional/traits/is_placeholder.hpp"
  SOURCES ${functional_sources}
  HEADERS ${functional_headers}
  COMPAT_HEADERS ${functional_compat_headers}
  MODULE_DEPENDENCIES
    hpx_assertion
    hpx_concurrency
    hpx_config
    hpx_datastructures
    hpx_debugging
    hpx_errors
    hpx_itt_notify
    hpx_serialization
    hpx_type_support
  CMAKE_SUBDIRS examples tests
)
