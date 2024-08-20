//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/allocator_support/traits/is_allocator.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/execution/algorithms/detail/inject_scheduler.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution/algorithms/run_loop.hpp>
#include <hpx/execution/algorithms/split.hpp>
#include <hpx/execution_base/completion_scheduler.hpp>
#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/functional/detail/tag_priority_invoke.hpp>

#include <exception>
#include <memory>
#include <utility>

namespace hpx::execution::experimental {

    // execution::ensure_started is used to eagerly start the execution of a
    // sender, while also providing a way to attach further work to execute once
    // it has completed.
    //
    // Once ensure_started returns, it is known that the provided sender has
    // been connected and start has been called on the resulting operation state
    // (see 5.2 Operation states represent work); in other words, the work
    // described by the provided sender has been submitted for execution on the
    // appropriate execution contexts. Returns a sender which completes when the
    // provided sender completes and sends values equivalent to those of the
    // provided sender.
    //
    // If the returned sender is destroyed before execution::connect() is
    // called, or if execution::connect() is called but the returned
    // operation-state is destroyed before execution::start() is called, then a
    // stop-request is sent to the eagerly launched operation and the operation
    // is detached and will run to completion in the background. Its result will
    // be discarded when it eventually completes.
    //
    // Note that the application will need to make sure that resources are kept
    // alive in the case that the operation detaches. e.g. by holding a
    // std::shared_ptr to those resources or otherwise having some out-of-band
    // way to signal completion of the operation so that resource release can be
    // sequenced after the completion.
    //
    inline constexpr struct ensure_started_t final
      : hpx::functional::detail::tag_priority<ensure_started_t>
    {
    private:
        // clang-format off
        template <typename Sender,
            typename Allocator = hpx::util::internal_allocator<>,
            HPX_CONCEPT_REQUIRES_(
                is_sender_v<Sender> &&
                hpx::traits::is_allocator_v<Allocator> &&
                experimental::detail::is_completion_scheduler_tag_invocable_v<
                    hpx::execution::experimental::set_value_t,
                    Sender, ensure_started_t, Allocator
                >
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_override_invoke(
            ensure_started_t, Sender&& sender, Allocator const& allocator = {})
        {
            auto scheduler =
                hpx::execution::experimental::get_completion_scheduler<
                    hpx::execution::experimental::set_value_t>(sender);

            return hpx::functional::tag_invoke(ensure_started_t{},
                HPX_MOVE(scheduler), HPX_FORWARD(Sender, sender), allocator);
        }

        // clang-format off
        template <typename Sender,
            typename Allocator = hpx::util::internal_allocator<>,
            HPX_CONCEPT_REQUIRES_(
                hpx::execution::experimental::is_sender_v<Sender> &&
                hpx::traits::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_invoke(ensure_started_t,
            hpx::execution::experimental::run_loop_scheduler const& sched,
            Sender&& sender, Allocator const& allocator = {})
        {
            auto split_sender = detail::split_sender<Sender, Allocator,
                detail::submission_type::eager,
                hpx::execution::experimental::run_loop_scheduler>{
                HPX_FORWARD(Sender, sender), allocator, sched};

            sched.get_run_loop().run();
            return split_sender;
        }

        // clang-format off
        template <typename Sender,
            typename Allocator = hpx::util::internal_allocator<>,
            HPX_CONCEPT_REQUIRES_(
                is_sender_v<Sender> &&
                hpx::traits::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            ensure_started_t, Sender&& sender, Allocator const& allocator = {})
        {
            return detail::split_sender<Sender, Allocator,
                detail::submission_type::eager>{
                HPX_FORWARD(Sender, sender), allocator};
        }

        template <typename Sender, typename Allocator>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            ensure_started_t,
            detail::split_sender<Sender, Allocator,
                detail::submission_type::eager>
                sender,
            Allocator const& = {})
        {
            return sender;
        }

        // clang-format off
        template <typename Scheduler, typename Allocator,
            HPX_CONCEPT_REQUIRES_(
                hpx::execution::experimental::is_scheduler_v<Scheduler> &&
                hpx::traits::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            ensure_started_t, Scheduler&& scheduler,
            Allocator const& allocator = {})
        {
            return hpx::execution::experimental::detail::inject_scheduler<
                ensure_started_t, Scheduler, Allocator>{
                HPX_FORWARD(Scheduler, scheduler), allocator};
        }

        // clang-format off
        template <typename Allocator = hpx::util::internal_allocator<>,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            ensure_started_t, Allocator const& allocator = {})
        {
            return detail::partial_algorithm<ensure_started_t, Allocator>{
                allocator};
        }
    } ensure_started{};
}    // namespace hpx::execution::experimental
