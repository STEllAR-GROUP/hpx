//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2022-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/modules/type_support.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx::execution::experimental::detail {

    // This is a partial s/r algorithm that injects a given scheduler as the
    // first argument while tag-invoking the bound algorithm.
    template <typename Tag, typename Scheduler, typename... Ts>
    struct inject_scheduler
      : partial_algorithm_base<Tag, hpx::util::make_index_pack_t<sizeof...(Ts)>,
            Ts...>
    {
    private:
        std::decay_t<Scheduler> scheduler;

        using base_type = partial_algorithm_base<Tag,
            hpx::util::make_index_pack_t<sizeof...(Ts)>, Ts...>;

    public:
        template <typename Scheduler_, typename... Ts_>
        // clang-format off
            requires (
                hpx::execution::experimental::is_scheduler_v<Scheduler_>
            )
        // clang-format on
        explicit constexpr inject_scheduler(Scheduler_&& scheduler, Ts_&&... ts)
          : base_type(HPX_FORWARD(Ts_, ts)...)
          , scheduler(HPX_FORWARD(Scheduler_, scheduler))
        {
        }

        template <typename U> 
        // clang-format off
            requires (
                hpx::execution::experimental::is_sender_v<U>
            )
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto operator|(
            U&& u, inject_scheduler p)
        {
            // NOLINTNEXTLINE(bugprone-use-after-move)
            return HPX_MOVE(p).invoke(HPX_MOVE(p.scheduler), HPX_FORWARD(U, u));
        }
    };
}    // namespace hpx::execution::experimental::detail
