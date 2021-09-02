//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/tag_dispatch.hpp>

#include <type_traits>

namespace hpx::execution::experimental {
    template <typename Scheduler>
    struct get_completion_scheduler_t final
      : hpx::functional::tag<get_completion_scheduler_t<Scheduler>>
    {
    };

    template <typename Scheduler>
    HPX_INLINE_CONSTEXPR_VARIABLE get_completion_scheduler_t<Scheduler>
        get_completion_scheduler{};

    namespace detail {
        template <bool TagDispatchable, typename CPO, typename Sender>
        struct has_completion_scheduler_impl : std::false_type
        {
        };

        template <typename CPO, typename Sender>
        struct has_completion_scheduler_impl<true, CPO, Sender>
          : hpx::execution::experimental::is_scheduler<hpx::functional::
                    tag_dispatch_result_t<get_completion_scheduler_t<CPO>,
                        std::decay_t<Sender> const&>>
        {
        };

        template <typename CPO, typename Sender>
        struct has_completion_scheduler
          : has_completion_scheduler_impl<
                hpx::functional::is_tag_dispatchable_v<
                    get_completion_scheduler_t<CPO>,
                    std::decay_t<Sender> const&>,
                CPO, Sender>
        {
        };

        template <typename CPO, typename Sender>
        HPX_INLINE_CONSTEXPR_VARIABLE bool has_completion_scheduler_v =
            has_completion_scheduler<CPO, Sender>::value;

        template <bool HasCompletionScheduler, typename ReceiverCPO,
            typename Sender, typename AlgorithmCPO, typename... Ts>
        struct is_completion_scheduler_tag_dispatchable_impl : std::false_type
        {
        };

        template <typename ReceiverCPO, typename Sender, typename AlgorithmCPO,
            typename... Ts>
        struct is_completion_scheduler_tag_dispatchable_impl<true, ReceiverCPO,
            Sender, AlgorithmCPO, Ts...>
          : std::integral_constant<bool,
                hpx::functional::is_tag_dispatchable_v<AlgorithmCPO,
                    hpx::functional::tag_dispatch_result_t<
                        hpx::execution::experimental::
                            get_completion_scheduler_t<ReceiverCPO>,
                        Sender>,
                    Sender, Ts...>>
        {
        };

        template <typename ReceiverCPO, typename Sender, typename AlgorithmCPO,
            typename... Ts>
        struct is_completion_scheduler_tag_dispatchable
          : is_completion_scheduler_tag_dispatchable_impl<
                hpx::execution::experimental::detail::
                    has_completion_scheduler_v<ReceiverCPO, Sender>,
                ReceiverCPO, Sender, AlgorithmCPO, Ts...>
        {
        };

        template <typename ReceiverCPO, typename Sender, typename AlgorithmCPO,
            typename... Ts>
        HPX_INLINE_CONSTEXPR_VARIABLE bool
            is_completion_scheduler_tag_dispatchable_v =
                is_completion_scheduler_tag_dispatchable<ReceiverCPO, Sender,
                    AlgorithmCPO, Ts...>::value;

    }    // namespace detail
}    // namespace hpx::execution::experimental
