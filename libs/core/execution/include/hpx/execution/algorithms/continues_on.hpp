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

#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution/algorithms/schedule_from.hpp>
#include <hpx/modules/concepts.hpp>
#include <hpx/modules/execution_base.hpp>
#include <hpx/modules/tag_invoke.hpp>

#include <utility>

namespace hpx::execution::experimental {

    inline constexpr struct continues_on_t final
      : hpx::functional::detail::tag_priority<continues_on_t>
    {
    private:
        template <typename Sender, typename Scheduler,
            HPX_CONCEPT_REQUIRES_(
                is_sender_v<Sender>&& is_scheduler_v<Scheduler>&& experimental::
                    detail::is_completion_scheduler_tag_invocable_v<
                        hpx::execution::experimental::set_value_t, Sender,
                        continues_on_t, Scheduler>)>
        friend constexpr HPX_FORCEINLINE auto tag_override_invoke(
            continues_on_t, Sender&& sender, Scheduler&& scheduler)
        {
            auto completion_scheduler =
                hpx::execution::experimental::get_completion_scheduler<
                    hpx::execution::experimental::set_value_t>(sender);

            return hpx::functional::tag_invoke(continues_on_t{},
                HPX_MOVE(completion_scheduler), HPX_FORWARD(Sender, sender),
                HPX_FORWARD(Scheduler, scheduler));
        }

        template <typename Sender, typename Scheduler,
            HPX_CONCEPT_REQUIRES_(
                is_sender_v<Sender>&& is_scheduler_v<Scheduler>)>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            continues_on_t, Sender&& predecessor_sender, Scheduler&& scheduler)
        {
            return schedule_from(HPX_FORWARD(Scheduler, scheduler),
                HPX_FORWARD(Sender, predecessor_sender));
        }

        template <typename Scheduler>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            continues_on_t, Scheduler&& scheduler)
        {
            return detail::partial_algorithm<continues_on_t, Scheduler>{
                HPX_FORWARD(Scheduler, scheduler)};
        }
    } continues_on{};
}    // namespace hpx::execution::experimental

#endif
