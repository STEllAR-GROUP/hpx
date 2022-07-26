//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/concepts/concepts.hpp>
#include <hpx/datastructures/member_pack.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/type_support/pack.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx::execution::experimental::detail {

    // This is a partial s/r algorithm that injects a given scheduler as the
    // first argument while tag-invoking the bound algorithm.
    template <typename Tag, typename Scheduler, typename IsPack, typename... Ts>
    struct inject_scheduler_base;

    template <typename Tag, typename Scheduler, std::size_t... Is,
        typename... Ts>
    struct inject_scheduler_base<Tag, Scheduler, hpx::util::index_pack<Is...>,
        Ts...>
    {
    private:
        std::decay_t<Scheduler> scheduler;
        HPX_NO_UNIQUE_ADDRESS hpx::util::member_pack_for<Ts...> ts;

    public:
        // clang-format off
        template <typename Scheduler_, typename... Ts_,
            HPX_CONCEPT_REQUIRES_(
                hpx::execution::experimental::is_scheduler_v<Scheduler_>
            )>
        // clang-format on
        explicit constexpr inject_scheduler_base(
            Scheduler_&& scheduler, Ts_&&... ts)
          : scheduler(HPX_FORWARD(Scheduler_, scheduler))
          , ts(std::piecewise_construct, HPX_FORWARD(Ts_, ts)...)
        {
        }

        inject_scheduler_base(inject_scheduler_base&&) = default;
        inject_scheduler_base& operator=(inject_scheduler_base&&) = default;
        inject_scheduler_base(inject_scheduler_base const&) = delete;
        inject_scheduler_base& operator=(inject_scheduler_base const&) = delete;

        // clang-format off
        template <typename U,
            HPX_CONCEPT_REQUIRES_(
                hpx::execution::experimental::is_sender_v<U>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto operator|(
            U&& u, inject_scheduler_base p)
        {
            return Tag{}(HPX_MOVE(p.scheduler), HPX_FORWARD(U, u),
                HPX_MOVE(p.ts).template get<Is>()...);
        }
    };

    template <typename Tag, typename Scheduler, typename... Ts>
    using inject_scheduler = inject_scheduler_base<Tag, Scheduler,
        hpx::util::make_index_pack_t<sizeof...(Ts)>, Ts...>;
}    // namespace hpx::execution::experimental::detail
