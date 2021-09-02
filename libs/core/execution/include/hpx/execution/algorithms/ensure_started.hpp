//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/allocator_support/traits/is_allocator.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution/algorithms/split.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>

#include <utility>

namespace hpx::execution::experimental {
    HPX_INLINE_CONSTEXPR_VARIABLE struct ensure_started_t final
      : hpx::functional::tag_fallback<ensure_started_t>
    {
    private:
        // clang-format off
        template <typename Sender,
            typename Allocator = hpx::util::internal_allocator<>,
            HPX_CONCEPT_REQUIRES_(
                is_sender_v<Sender> &&
                hpx::traits::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            ensure_started_t, Sender&& sender, Allocator const& allocator = {})
        {
            return detail::split_sender<Sender, Allocator,
                detail::submission_type::eager>{
                std::forward<Sender>(sender), allocator};
        }

        template <typename Sender, typename Allocator>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            ensure_started_t,
            detail::split_sender<Sender, Allocator,
                detail::submission_type::eager>
                sender,
            Allocator const& = {})
        {
            return sender;
        }

        // clang-format off
        template <typename Allocator = hpx::util::internal_allocator<>,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            ensure_started_t, Allocator const& allocator = {})
        {
            return detail::partial_algorithm<ensure_started_t, Allocator>{
                allocator};
        }
    } ensure_started{};
}    // namespace hpx::execution::experimental
