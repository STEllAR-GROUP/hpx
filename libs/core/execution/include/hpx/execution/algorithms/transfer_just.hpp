//  Copyright (c) 2020 ETH Zurich
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

namespace hpx { namespace execution { namespace experimental {
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
}}}    // namespace hpx::execution::experimental
