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

#include <hpx/execution/algorithms/transfer.hpp>
#include <hpx/execution/algorithms/when_all_with_variant.hpp>
#include <hpx/modules/concepts.hpp>
#include <hpx/modules/execution_base.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/tag_invoke.hpp>

#include <utility>

namespace hpx::execution::experimental {

    inline constexpr struct transfer_when_all_with_variant_t final
      : hpx::functional::detail::tag_fallback<transfer_when_all_with_variant_t>
    {
    private:
        template <typename Sched, typename... Senders,
            HPX_CONCEPT_REQUIRES_(is_scheduler_v<Sched>&&
                    hpx::util::all_of_v<is_sender<Senders>...>)>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            transfer_when_all_with_variant_t, Sched&& sched,
            Senders&&... senders)
        {
            return transfer(
                when_all_with_variant(HPX_FORWARD(Senders, senders)...),
                HPX_FORWARD(Sched, sched));
        }
    } transfer_when_all_with_variant{};
}    // namespace hpx::execution::experimental

#endif
