//  Copyright (c) 2012 Thomas Heller
//  Copyright (c) 2013-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <functional>
#include <type_traits>
#include <utility>

namespace hpx::util {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        template <typename TD>
        struct decay_unwrap_impl
        {
            using type = TD;

            template <typename T>
            constexpr static T call(T&& t) noexcept
            {
                return std::forward<T>(t);
            }
        };

        template <typename X>
        struct decay_unwrap_impl<::std::reference_wrapper<X>>
        {
            using type = X&;

            constexpr static decltype(auto) call(
                ::std::reference_wrapper<X> ref) noexcept
            {
                return ref.get();
            }
        };
    }    // namespace detail

    template <typename T>
    struct decay_unwrap : detail::decay_unwrap_impl<std::decay_t<T>>
    {
    };

    template <typename T>
    using decay_unwrap_t = typename decay_unwrap<T>::type;
}    // namespace hpx::util
