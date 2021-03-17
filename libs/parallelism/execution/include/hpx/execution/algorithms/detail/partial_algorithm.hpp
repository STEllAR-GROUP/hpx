//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <type_traits>
#include <utility>

namespace hpx {
    namespace execution {
        namespace experimental {
            namespace detail {
    template <typename Tag, typename... Ts>
    struct partial_algorithm;

    template <typename Tag, typename T>
    struct partial_algorithm<Tag, T>
    {
        std::decay_t<T> t;

        template <typename U>
        friend constexpr HPX_FORCEINLINE auto operator|(
            U&& u, partial_algorithm p)
        {
            return Tag{}(std::forward<U>(u), std::move(p.t));
        }
    };

    template <typename Tag>
    struct partial_algorithm<Tag>
    {
        template <typename U>
        friend constexpr HPX_FORCEINLINE auto operator|(
            U&& u, partial_algorithm)
        {
            return Tag{}(std::forward<U>(u));
        }
    };
}}}}    // namespace hpx::execution::experimental::detail
