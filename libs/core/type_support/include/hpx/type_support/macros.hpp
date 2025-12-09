//  Copyright (c) 2013 Agustin Berge
//  Copyright (c) 2022-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config/export_definitions.hpp>

#if !defined(HPX_COMPILE_WITH_MODULES) || defined(HPX_COMPILE_BMI)
#include <hpx/config.hpp>
#include <hpx/type_support/assert_owns_lock.hpp>
#include <hpx/type_support/is_trivially_relocatable.hpp>
#include <hpx/type_support/unused.hpp>
#if defined(HPX_HAVE_CXX20_COROUTINES)
#include <hpx/type_support/coroutines_support.hpp>
#endif

#include <type_traits>
#endif

//////////////////////////////////////////////////////////////////////////////
// from hpx/type_support/assert_owns_lock.hpp
#define HPX_ASSERT_OWNS_LOCK(l) ::hpx::util::detail::assert_owns_lock(l, 0L)

#define HPX_ASSERT_DOESNT_OWN_LOCK(l)                                          \
    ::hpx::util::detail::assert_doesnt_own_lock(l, 0L)

//////////////////////////////////////////////////////////////////////////////
// from hpx/type_support/is_trivially_relocatable.hpp
// Macro to specialize template for given type
#define HPX_DECLARE_TRIVIALLY_RELOCATABLE(T)                                   \
    namespace hpx::experimental {                                              \
        template <>                                                            \
        struct is_trivially_relocatable<T> : std::true_type                    \
        {                                                                      \
        };                                                                     \
    }

#define HPX_DECLARE_TRIVIALLY_RELOCATABLE_TEMPLATE(T)                          \
    namespace hpx::experimental {                                              \
        template <typename... K>                                               \
        struct is_trivially_relocatable<T<K...>> : std::true_type              \
        {                                                                      \
        };                                                                     \
    }

#define HPX_DECLARE_TRIVIALLY_RELOCATABLE_TEMPLATE_IF(T, Condition)            \
    namespace hpx::experimental {                                              \
        template <typename... K>                                               \
        struct is_trivially_relocatable<T<K...>> : Condition<K...>             \
        {                                                                      \
        };                                                                     \
    }

//////////////////////////////////////////////////////////////////////////////
// from hpx/type_support/unused.hpp
// use this to silence compiler warnings related to unused function arguments.
#if defined(__CUDA_ARCH__)
#define HPX_UNUSED(x) (void) x
#else
#define HPX_UNUSED(x) ::hpx::util::unused = (x)
#endif

//////////////////////////////////////////////////////////////////////////////
// from hpx/type_support/coroutine_support.hpp
#if defined(HPX_HAVE_CXX20_COROUTINES)

#if defined(__has_include)
#if __has_include(<coroutine>)

#define HPX_COROUTINE_NAMESPACE_STD std

#elif __has_include(<experimental/coroutine>)

#define HPX_COROUTINE_NAMESPACE_STD std::experimental

#endif
#endif

#if !defined(HPX_COROUTINE_NAMESPACE_STD)
#error "Platform does not support C++20 coroutines"
#endif

#endif    // HPX_HAVE_CXX20_COROUTINES
