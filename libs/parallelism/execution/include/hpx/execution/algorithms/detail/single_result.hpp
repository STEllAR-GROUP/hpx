//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/type_support/pack.hpp>

namespace hpx {
    namespace execution {
        namespace experimental {
            namespace detail {
    template <typename Variants>
    struct sync_wait_single_result
    {
        static_assert(sizeof(Variants) == 0,
            "sync_wait expects the predecessor sender to have a single variant "
            "with a single type in sender_traits<>::value_types");
    };

    template <>
    struct sync_wait_single_result<hpx::util::pack<hpx::util::pack<>>>
    {
        using type = void;
    };

    template <typename T>
    struct sync_wait_single_result<hpx::util::pack<hpx::util::pack<T>>>
    {
        using type = T;
    };

    template <typename T, typename U, typename... Ts>
    struct sync_wait_single_result<
        hpx::util::pack<hpx::util::pack<T, U, Ts...>>>
    {
        static_assert(sizeof(T) == 0,
            "sync_wait expects the predecessor sender to have a single variant "
            "with a single type in sender_traits<>::value_types (single "
            "variant with two or more types given)");
    };

    template <typename T, typename U, typename... Ts>
    struct sync_wait_single_result<hpx::util::pack<T, U, Ts...>>
    {
        static_assert(sizeof(T) == 0,
            "sync_wait expects the predecessor sender to have a single variant "
            "with a single type in sender_traits<>::value_types (two or more "
            "variants given)");
    };
}}}}    // namespace hpx::execution::experimental::detail
