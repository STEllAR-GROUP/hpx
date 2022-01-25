//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/tag_invoke.hpp>

#include <type_traits>

namespace hpx::execution::experimental {

    // execution::get_completion_scheduler is used to ask a sender object for
    // the completion scheduler for one of its signals.
    //
    // The name execution::get_completion_scheduler denotes a customization
    // point object template. For some subexpression s, let S be decltype((s)).
    // If S does not satisfy execution::sender,
    // execution::get_completion_scheduler<CPO>(s) is ill-formed for all
    // template arguments CPO. If the template argument CPO in
    // execution::get_completion_scheduler<CPO> is not one of
    // execution::set_value_t, execution::set_error_t, or
    // execution::set_stopped_t, execution::get_completion_scheduler<CPO> is
    // ill-formed. Otherwise, execution::get_completion_scheduler<CPO>(s) is
    // expression-equivalent to:
    //
    //    1. tag_invoke(execution::get_completion_scheduler<CPO>, as_const(s))
    //       if this expression is well formed.
    //
    //       - Mandates: The tag_invoke expression above is not potentially
    //         throwing and its type satisfies execution::scheduler.
    //
    //    2.   Otherwise, execution::get_completion_scheduler<CPO>(s) is
    //         ill-formed.
    //
    // clang-format off
    template <typename CPO,
        HPX_CONCEPT_REQUIRES_(
            std::is_same_v<CPO, set_value_t> ||
            std::is_same_v<CPO, set_error_t> ||
            std::is_same_v<CPO, set_stopped_t>
        )>
    // clang-format on
    struct get_completion_scheduler_t final
      : hpx::functional::tag<get_completion_scheduler_t<CPO>>
    {
    };

    template <typename CPO>
    inline constexpr get_completion_scheduler_t<CPO> get_completion_scheduler{};

    namespace detail {
        template <bool TagInvocable, typename CPO, typename Sender>
        struct has_completion_scheduler_impl : std::false_type
        {
        };

        template <typename CPO, typename Sender>
        struct has_completion_scheduler_impl<true, CPO, Sender>
          : hpx::execution::experimental::is_scheduler<hpx::functional::
                    tag_invoke_result_t<get_completion_scheduler_t<CPO>,
                        std::decay_t<Sender> const&>>
        {
        };

        template <typename CPO, typename Sender>
        struct has_completion_scheduler
          : has_completion_scheduler_impl<hpx::functional::is_tag_invocable_v<
                                              get_completion_scheduler_t<CPO>,
                                              std::decay_t<Sender> const&>,
                CPO, Sender>
        {
        };

        template <typename CPO, typename Sender>
        inline constexpr bool has_completion_scheduler_v =
            has_completion_scheduler<CPO, Sender>::value;

        template <bool HasCompletionScheduler, typename ReceiverCPO,
            typename Sender, typename AlgorithmCPO, typename... Ts>
        struct is_completion_scheduler_tag_invocable_impl : std::false_type
        {
        };

        template <typename ReceiverCPO, typename Sender, typename AlgorithmCPO,
            typename... Ts>
        struct is_completion_scheduler_tag_invocable_impl<true, ReceiverCPO,
            Sender, AlgorithmCPO, Ts...>
          : std::integral_constant<bool,
                hpx::functional::is_tag_invocable_v<AlgorithmCPO,
                    hpx::functional::tag_invoke_result_t<
                        hpx::execution::experimental::
                            get_completion_scheduler_t<ReceiverCPO>,
                        Sender>,
                    Sender, Ts...>>
        {
        };

        template <typename ReceiverCPO, typename Sender, typename AlgorithmCPO,
            typename... Ts>
        struct is_completion_scheduler_tag_invocable
          : is_completion_scheduler_tag_invocable_impl<
                hpx::execution::experimental::detail::
                    has_completion_scheduler_v<ReceiverCPO, Sender>,
                ReceiverCPO, Sender, AlgorithmCPO, Ts...>
        {
        };

        template <typename ReceiverCPO, typename Sender, typename AlgorithmCPO,
            typename... Ts>
        inline constexpr bool is_completion_scheduler_tag_invocable_v =
            is_completion_scheduler_tag_invocable<ReceiverCPO, Sender,
                AlgorithmCPO, Ts...>::value;

    }    // namespace detail
}    // namespace hpx::execution::experimental
