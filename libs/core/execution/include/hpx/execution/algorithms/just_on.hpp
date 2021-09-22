//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/algorithms/just.hpp>
#include <hpx/execution/algorithms/on.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>

#include <utility>

namespace hpx { namespace execution { namespace experimental {
    HPX_INLINE_CONSTEXPR_VARIABLE struct just_on_t final
      : hpx::functional::tag_fallback<just_on_t>
    {
    private:
        template <typename Scheduler, typename... Ts>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            just_on_t, Scheduler&& scheduler, Ts&&... ts)
        {
            return on(just(std::forward<Ts>(ts)...),
                std::forward<Scheduler>(scheduler));
        }
    } just_on{};
}}}    // namespace hpx::execution::experimental
