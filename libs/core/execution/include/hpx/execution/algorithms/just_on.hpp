//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/algorithms/just.hpp>
#include <hpx/execution/algorithms/on.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>

#include <utility>

namespace hpx { namespace execution { namespace experimental {
    inline constexpr struct just_on_t final
      : hpx::functional::detail::tag_fallback<just_on_t>
    {
    private:
        template <typename Scheduler, typename... Ts>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            just_on_t, Scheduler&& scheduler, Ts&&... ts)
        {
            return on(just(HPX_FORWARD(Ts, ts)...),
                HPX_FORWARD(Scheduler, scheduler));
        }
    } just_on{};
}}}    // namespace hpx::execution::experimental
