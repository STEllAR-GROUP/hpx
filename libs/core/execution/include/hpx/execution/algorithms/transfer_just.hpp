//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/algorithms/just.hpp>
#include <hpx/execution/algorithms/transfer.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>

#include <utility>

namespace hpx::execution::experimental {

    // execution::transfer_just is used to create a sender that propagates a set
    // of values to a connected receiver on an execution agent belonging to the
    // associated execution context of a specified scheduler.
    //
    // Returns a sender whose value completion scheduler is the provided
    // scheduler, which sends the provided values in the same manner as just.
    //
    // This adaptor is provided as it greatly simplifies lifting values into
    // senders.
    inline constexpr struct transfer_just_t final
      : hpx::functional::detail::tag_fallback<transfer_just_t>
    {
    private:
        template <typename Scheduler, typename... Ts>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            transfer_just_t, Scheduler&& scheduler, Ts&&... ts)
        {
            return transfer(just(HPX_FORWARD(Ts, ts)...),
                HPX_FORWARD(Scheduler, scheduler));
        }
    } transfer_just{};
}    // namespace hpx::execution::experimental
