//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/execution/algorithms/detach.hpp>
#include <hpx/execution/algorithms/transform.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>

#include <utility>

namespace hpx { namespace execution { namespace experimental {
    HPX_INLINE_CONSTEXPR_VARIABLE struct execute_t final
      : hpx::functional::tag_fallback<execute_t>
    {
    private:
        template <typename Scheduler, typename F>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            execute_t, Scheduler&& scheduler, F&& f)
        {
            return detach(
                transform(schedule(std::forward<Scheduler>(scheduler)),
                    std::forward<F>(f)));
        }
    } execute{};
}}}    // namespace hpx::execution::experimental
