//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/functional/traits/is_invocable.hpp>

#include <type_traits>
#include <utility>

namespace hpx::experimental {

    inline constexpr struct prefer_t
      : hpx::functional::detail::tag_fallback<prefer_t>
    {
        // clang-format off
        template <typename Tag, typename... Tn>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
                prefer_t, Tag tag, Tn&&... tn)
            noexcept(noexcept(tag(HPX_FORWARD(Tn, tn)...)))
            -> decltype(tag(HPX_FORWARD(Tn, tn)...))
        // clang-format on
        {
            return tag(HPX_FORWARD(Tn, tn)...);
        }

        // clang-format off
        template <typename Tag, typename T0, typename... Tn>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
                prefer_t, Tag, T0&& t0, Tn&&...)
            noexcept(noexcept(HPX_FORWARD(T0, t0)))
            -> std::enable_if_t<
                    !hpx::functional::is_tag_invocable_v<
                        prefer_t, Tag, T0, Tn...> &&
                    !hpx::is_invocable_v<Tag, T0, Tn...>,
                    decltype(HPX_FORWARD(T0, t0))>
        // clang-format on
        {
            return HPX_FORWARD(T0, t0);
        }
    } prefer{};
}    // namespace hpx::experimental
