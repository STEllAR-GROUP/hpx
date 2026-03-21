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

#include <hpx/execution/algorithms/into_variant.hpp>
#include <hpx/execution/algorithms/when_all.hpp>
#include <hpx/modules/concepts.hpp>
#include <hpx/modules/execution_base.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/tag_invoke.hpp>

#include <utility>

namespace hpx::execution::experimental {

    inline constexpr struct when_all_with_variant_t final
      : hpx::functional::detail::tag_fallback<when_all_with_variant_t>
    {
    private:
        template <typename... Senders,
            HPX_CONCEPT_REQUIRES_((sizeof...(Senders) > 0) &&
                hpx::util::all_of_v<is_sender<Senders>...>)>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            when_all_with_variant_t, Senders&&... senders)
        {
            return when_all(into_variant(HPX_FORWARD(Senders, senders))...);
        }
    } when_all_with_variant{};
}    // namespace hpx::execution::experimental

#endif
