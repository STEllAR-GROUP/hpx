//  Copyright (c) 2022-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/concepts/concepts.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/executors/explicit_scheduler_executor.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>

#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

    namespace detail {

        template <typename Scheduler, typename Enable = void>
        struct exposes_policy_aware_scheduler_types : std::false_type
        {
        };

        template <typename Scheduler>
        struct exposes_policy_aware_scheduler_types<Scheduler,
            std::void_t<typename Scheduler::policy_type,
                typename Scheduler::base_scheduler_type>> : std::true_type
        {
        };

        template <typename Scheduler, typename Enable = void>
        struct exposes_get_policy : std::false_type
        {
        };

        // clang-format off
        template <typename Scheduler>
        struct exposes_get_policy<Scheduler,
            std::enable_if_t<hpx::is_execution_policy_v<
                decltype(std::declval<Scheduler>().get_policy())>>>
          : std::true_type
        {
        };
        // clang-format on
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler, typename ExPolicy>
    struct scheduler_and_policy : std::decay_t<Scheduler>
    {
        using base_scheduler_type = std::decay_t<Scheduler>;
        using policy_type = std::decay_t<ExPolicy>;

        template <typename Scheduler_, typename ExPolicy_>
        scheduler_and_policy(Scheduler_&& sched, ExPolicy_&& policy)
          : base_scheduler_type(HPX_FORWARD(Scheduler_, sched))
          , policy(HPX_FORWARD(ExPolicy_, policy))
        {
        }

        constexpr policy_type const& get_policy() const noexcept
        {
            return policy;
        }

        constexpr base_scheduler_type const& get_scheduler() const noexcept
        {
            return static_cast<base_scheduler_type const&>(*this);
        }

        policy_type policy;
    };

    // different versions of clang-format disagree
    // clang-format off
    template <typename Scheduler, typename ExPolicy>
    scheduler_and_policy(Scheduler&&, ExPolicy&&)
        -> scheduler_and_policy<std::decay_t<Scheduler>,
            std::decay_t<ExPolicy>>;
    // clang-format on

    ////////////////////////////////////////////////////////////////////////////
    // support all scheduling properties exposed by the embedded scheduler
    // clang-format off
    template <typename Tag, typename Scheduler, typename ExPolicy,
        typename Property,
        HPX_CONCEPT_REQUIRES_(
            hpx::execution::experimental::is_scheduling_property_v<Tag>
        )>
    auto tag_invoke(Tag tag,
        scheduler_and_policy<Scheduler, ExPolicy> const& scheduler,
        Property&& prop)
        -> decltype(scheduler_and_policy<Scheduler, ExPolicy>(
                std::declval<Tag>()(
                    std::declval<Scheduler>(), std::declval<Property>()),
                std::declval<ExPolicy>()))
    // clang-format on
    {
        return scheduler_and_policy<Scheduler, ExPolicy>(
            tag(scheduler.get_scheduler(), HPX_FORWARD(Property, prop)),
            scheduler.get_policy());
    }

    // clang-format off
    template <typename Tag, typename Scheduler, typename ExPolicy,
        HPX_CONCEPT_REQUIRES_(
            hpx::execution::experimental::is_scheduling_property_v<Tag>
        )>
    // clang-format on
    auto tag_invoke(
        Tag tag, scheduler_and_policy<Scheduler, ExPolicy> const& scheduler)
        -> decltype(std::declval<Tag>()(std::declval<Scheduler>()))
    {
        return tag(scheduler.get_scheduler());
    }

    // Experimental support for facilities from p2500 (wg21.link/p2500)
    inline namespace p2500 {

        ///////////////////////////////////////////////////////////////////////
        // policy_aware_scheduler is a concept for parallel algorithms that
        // represents a combined scheduler and execution_policy entity. It
        // allows to get both execution policy type and execution policy object
        // parallel algorithm is called with.
        //
        // Customizations of the parallel algorithms can reuse the existing
        // implementation of parallel algorithms with ExecutionPolicy template
        // parameter for "known" base_scheduler_type type.
        template <typename Scheduler, typename Enable = void>
        struct is_policy_aware_scheduler : std::false_type
        {
        };

        template <typename Scheduler>
        struct is_policy_aware_scheduler<Scheduler,
            std::enable_if_t<is_scheduler_v<Scheduler> &&
                detail::exposes_policy_aware_scheduler_types<
                    std::decay_t<Scheduler>>::value &&
                detail::exposes_get_policy<Scheduler>::value>> : std::true_type
        {
        };

        template <typename Scheduler>
        inline constexpr bool is_policy_aware_scheduler_v =
            is_policy_aware_scheduler<Scheduler>::value;

        ///////////////////////////////////////////////////////////////////////
        // execute_on is the customization point that serves the purpose to tie
        // scheduler and execution_policy.
        //
        // It's up to scheduler customization to check if it can work with the
        // passed execution policy.
        inline constexpr struct execute_on_t final
          : hpx::functional::detail::tag_fallback<execute_on_t>
        {
        private:
            // clang-format off
            template <typename Scheduler, typename ExPolicy,
                HPX_CONCEPT_REQUIRES_(
                    hpx::execution::experimental::is_scheduler_v<
                        std::decay_t<Scheduler>> &&
                    hpx::is_execution_policy_v<ExPolicy>
                )>
            // clang-format on
            friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
                execute_on_t, Scheduler&& scheduler, ExPolicy&& policy)
            {
                return scheduler_and_policy(HPX_FORWARD(Scheduler, scheduler),
                    HPX_FORWARD(ExPolicy, policy));
            }
        } execute_on{};
    }    // namespace p2500
}    // namespace hpx::execution::experimental
