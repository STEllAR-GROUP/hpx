//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

namespace hpx { namespace util {

    namespace detail {

        template <typename T>
        struct hide_from_adl
        {
            struct inner
            {
                using type = T;
            };
        };
    }    // namespace detail

    template <typename T>
    using hide_from_adl = typename detail::hide_from_adl<T>::inner;

    template <typename T>
    using hidden_from_adl_t = typename T::type;
}}    // namespace hpx::util
