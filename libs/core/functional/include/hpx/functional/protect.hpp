//  Copyright (c) 2002 Peter Dimov and Multi Media Ltd.
//  Copyright (c) 2009 Steven Watanabe
//  Copyright (c) 2011-2023 Hartmut Kaiser
//  Copyright (c) 2013-2016 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/traits/is_bind_expression.hpp>

#include <type_traits>
#include <utility>

namespace hpx::util {

    namespace detail {

        template <typename F>
        class protected_bind : public F
        {
        public:
            HPX_HOST_DEVICE explicit protected_bind(F const& f)
              : F(f)
            {
            }

            HPX_HOST_DEVICE explicit protected_bind(F&& f) noexcept
              : F(HPX_MOVE(f))
            {
            }

#if !defined(__NVCC__) && !defined(__CUDACC__)
            protected_bind(protected_bind const&) = default;
            protected_bind(protected_bind&&) = default;
#else
            HPX_HOST_DEVICE protected_bind(protected_bind const& other)
              : F(other)
            {
            }

            HPX_HOST_DEVICE protected_bind(protected_bind&& other) noexcept
              : F(HPX_MOVE(other))
            {
            }
#endif

            protected_bind& operator=(protected_bind const&) = default;
            protected_bind& operator=(protected_bind&&) = default;

            ~protected_bind() = default;
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    HPX_HOST_DEVICE std::enable_if_t<hpx::is_bind_expression_v<std::decay_t<T>>,
        detail::protected_bind<std::decay_t<T>>>
    protect(T&& f)
    {
        return detail::protected_bind<std::decay_t<T>>(HPX_FORWARD(T, f));
    }

    // leave everything that is not a bind expression as is
    template <typename T>
    HPX_HOST_DEVICE
        std::enable_if_t<!hpx::is_bind_expression_v<std::decay_t<T>>, T&&>
        protect(T&& v)    //-V659
    {
        return HPX_FORWARD(T, v);
    }
}    // namespace hpx::util
