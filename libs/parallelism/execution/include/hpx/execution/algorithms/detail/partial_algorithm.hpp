//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/datastructures/member_pack.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx {
    namespace execution {
        namespace experimental {
            namespace detail {
    template <typename Tag, typename IsPack, typename... Ts>
    struct partial_algorithm_base;

    template <typename Tag, std::size_t... Is, typename... Ts>
    struct partial_algorithm_base<Tag, hpx::util::index_pack<Is...>, Ts...>
    {
    private:
        hpx::util::member_pack_for<Ts...> ts;

    public:
        template <typename... Ts_>
        explicit constexpr partial_algorithm_base(Ts_&&... ts)
          : ts(std::piecewise_construct, std::forward<Ts_>(ts)...)
        {
        }

        partial_algorithm_base(partial_algorithm_base&&) = default;
        partial_algorithm_base& operator=(partial_algorithm_base&&) = default;
        partial_algorithm_base(partial_algorithm_base const&) = delete;
        partial_algorithm_base& operator=(
            partial_algorithm_base const&) = delete;

        template <typename U>
        friend constexpr HPX_FORCEINLINE auto operator|(
            U&& u, partial_algorithm_base p)
        {
            return Tag{}(
                std::forward<U>(u), std::move(p.ts).template get<Is>()...);
        }
    };

    template <typename Tag, typename... Ts>
    using partial_algorithm = partial_algorithm_base<Tag,
        typename hpx::util::make_index_pack<sizeof...(Ts)>::type, Ts...>;
}}}}    // namespace hpx::execution::experimental::detail
