//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/algorithms/start_detached.hpp>
#include <hpx/execution/algorithms/then.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>

#include <utility>

namespace hpx { namespace execution { namespace experimental {
    inline constexpr struct execute_t final
      : hpx::functional::detail::tag_fallback<execute_t>
    {
    private:
        template <typename Scheduler, typename F>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            execute_t, Scheduler&& scheduler, F&& f)
        {
            return start_detached(
                then(schedule(HPX_FORWARD(Scheduler, scheduler)),
                    HPX_FORWARD(F, f)));
        }
    } execute{};
}}}    // namespace hpx::execution::experimental
