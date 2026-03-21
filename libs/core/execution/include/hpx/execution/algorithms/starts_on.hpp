//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_STDEXEC)
#include <hpx/modules/execution_base.hpp>
#else

#include <hpx/execution/algorithms/detail/inject_scheduler.hpp>
#include <hpx/execution/algorithms/schedule_from.hpp>
#include <hpx/modules/concepts.hpp>
#include <hpx/modules/execution_base.hpp>
#include <hpx/modules/tag_invoke.hpp>

#include <utility>

namespace hpx::execution::experimental {

    inline constexpr struct starts_on_t final
      : hpx::functional::detail::tag_priority<starts_on_t>
    {
    private:
        template <typename Scheduler, typename Sender,
            HPX_CONCEPT_REQUIRES_(
                is_scheduler_v<Scheduler>&& is_sender_v<Sender>)>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            starts_on_t, Scheduler&& scheduler, Sender&& sender)
        {
            return schedule_from(
                HPX_FORWARD(Scheduler, scheduler), HPX_FORWARD(Sender, sender));
        }

        template <typename Scheduler,
            HPX_CONCEPT_REQUIRES_(is_scheduler_v<Scheduler>)>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            starts_on_t, Scheduler&& scheduler)
        {
            return detail::inject_scheduler<starts_on_t, Scheduler>{
                HPX_FORWARD(Scheduler, scheduler)};
        }
    } starts_on{};
}    // namespace hpx::execution::experimental

#endif
