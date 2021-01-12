//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/tag_fallback_invoke.hpp>
#include <hpx/functional/traits/is_invocable.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace experimental {
    HPX_INLINE_CONSTEXPR_VARIABLE struct prefer_t
      : hpx::functional::tag_fallback<prefer_t>
    {
        template <typename Tag, typename... Tn>
        friend constexpr HPX_FORCEINLINE auto
        tag_fallback_invoke(prefer_t, Tag&& tag, Tn&&... tn) noexcept(
            noexcept(hpx::functional::tag_invoke(
                std::forward<Tag>(tag), std::forward<Tn>(tn)...)))
            -> decltype(hpx::functional::tag_invoke(
                std::forward<Tag>(tag), std::forward<Tn>(tn)...))
        {
            return hpx::functional::tag_invoke(
                std::forward<Tag>(tag), std::forward<Tn>(tn)...);
        }

        template <typename Tag, typename T0, typename... Tn>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(prefer_t,
            Tag&&, T0&& t0, Tn&&...) noexcept(noexcept(std::forward<T0>(t0))) ->
            typename std::enable_if<!hpx::functional::is_tag_invocable<prefer_t,
                                        Tag, T0, Tn...>::value &&
                    !hpx::functional::is_tag_invocable<Tag, T0, Tn...>::value,
                decltype(std::forward<T0>(t0))>::type
        {
            return std::forward<T0>(t0);
        }
    } prefer{};
}}    // namespace hpx::experimental
