//  Copyright (c) 2020 Hartmut Kaiser
//  Copyright 2020 Hana Duskov
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This was taken from Hana here:
// https://twitter.com/hankadusikova/status/1276828584179642368

#pragma once

#include <hpx/config.hpp>

namespace hpx { namespace util {

    namespace detail {

        template <typename F>
        struct noexcept_cast_helper;

        template <typename Ret, typename... Args>
        struct noexcept_cast_helper<Ret (*)(Args...)>
        {
            using type = Ret (*)(Args...) noexcept;
        };

#if defined(HPX_HAVE_CXX17_NOEXCEPT_FUNCTIONS_AS_NONTYPE_TEMPLATE_ARGUMENTS)
        template <typename Ret, typename... Args>
        struct noexcept_cast_helper<Ret (*)(Args...) noexcept>
        {
            using type = Ret (*)(Args...) noexcept;
        };
#endif
    }    // namespace detail

    template <typename T>
    auto noexcept_cast(T const obj) noexcept
    {
        return reinterpret_cast<typename detail::noexcept_cast_helper<T>::type>(
            obj);
    };
}}    // namespace hpx::util
