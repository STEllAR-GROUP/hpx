//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution/algorithms/schedule_from.hpp>
#include <hpx/execution_base/completion_scheduler.hpp>
#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/detail/tag_priority_invoke.hpp>

#include <utility>

namespace hpx::execution::experimental {

    // execution::transfer is used to adapt a sender into a sender with a
    // different associated set_value completion scheduler.
    //
    // [Note: it results in a transition between different execution contexts
    // when executed. --end note]
    //
    // Returns a sender describing the transition from the execution agent of
    // the input sender to the execution agent of the target scheduler.
    //
    // The execution::on algorithm will ensure that the given sender will start
    // in the specified context, and doesn't care where the completion signal
    // for that sender is sent. The execution::transfer algorithm will not care
    // where the given sender is going to be started, but will ensure that the
    // completion signal of will be transferred to the given context.
    //
    inline constexpr struct transfer_t final
      : hpx::functional::detail::tag_priority<transfer_t>
    {
    private:
        // clang-format off
        template <typename Sender, typename Scheduler,
            HPX_CONCEPT_REQUIRES_(
                is_sender_v<Sender> &&
                is_scheduler_v<Scheduler> &&
                experimental::detail::is_completion_scheduler_tag_invocable_v<
                    hpx::execution::experimental::set_value_t, Sender,
                    transfer_t, Scheduler
                >
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_override_invoke(
            transfer_t, Sender&& sender, Scheduler&& scheduler)
        {
            auto completion_scheduler =
                hpx::execution::experimental::get_completion_scheduler<
                    hpx::execution::experimental::set_value_t>(sender);

            return hpx::functional::tag_invoke(transfer_t{},
                HPX_MOVE(completion_scheduler), HPX_FORWARD(Sender, sender),
                HPX_FORWARD(Scheduler, scheduler));
        }

        // clang-format off
        template <typename Sender, typename Scheduler,
            HPX_CONCEPT_REQUIRES_(
                is_sender_v<Sender> &&
                is_scheduler_v<Scheduler>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            transfer_t, Sender&& predecessor_sender, Scheduler&& scheduler)
        {
            return schedule_from(HPX_FORWARD(Scheduler, scheduler),
                HPX_FORWARD(Sender, predecessor_sender));
        }

        template <typename Scheduler>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            transfer_t, Scheduler&& scheduler)
        {
            return detail::partial_algorithm<transfer_t, Scheduler>{
                HPX_FORWARD(Scheduler, scheduler)};
        }
    } transfer{};
}    // namespace hpx::execution::experimental
