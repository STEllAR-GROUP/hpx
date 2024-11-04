//  Copyright (c) 2024 Shreyas Atre
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/parallel/algorithms/detail/rfa.hpp>
#include <hpx/parallel/util/loop.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>
#include "rfa.hpp"

namespace hpx::parallel::detail {

    template <typename ExPolicy>
    struct sequential_reduce_deterministic_t final
      : hpx::functional::detail::tag_fallback<
            sequential_reduce_deterministic_t<ExPolicy>>
    {
    private:
        template <typename InIterB, typename InIterE, typename T,
            typename Reduce>
        friend constexpr T tag_fallback_invoke(
            sequential_reduce_deterministic_t, ExPolicy&&, InIterB first,
            InIterE last, T init, Reduce&& r)
        {
            RFA::ReproducibleFloatingAccumulator<T> rfa;
            rfa += init;
            rfa.add(first, last);
            return rfa.conv();
        }
    };

#if !defined(HPX_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    inline constexpr sequential_reduce_deterministic_t<ExPolicy>
        sequential_reduce_deterministic =
            sequential_reduce_deterministic_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename... Args>
    HPX_HOST_DEVICE HPX_FORCEINLINE auto sequential_reduce_deterministic(
        Args&&... args)
    {
        return sequential_reduce_deterministic_t<ExPolicy>{}(
            std::forward<Args>(args)...);
    }
#endif

}    // namespace hpx::parallel::detail
