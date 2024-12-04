//  Copyright (c) 2024 Shreyas Atre
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/parallel/algorithms/detail/rfa_cuda.hpp>
#include <hpx/parallel/util/loop.hpp>

#include <cstddef>
#include <limits>
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
            hpx::parallel::detail::rfa::RFA_bins<T> bins;
            bins.initialize_bins();
            memcpy(rfa::bin_host_buffer, &bins, sizeof(bins));

            hpx::parallel::detail::rfa::ReproducibleFloatingAccumulator<T> rfa;
            rfa.set_max_abs_val(init);
            rfa.unsafe_add(init);
            rfa.renorm();
            // rfa.add(first, last);
            for (auto e = first; e != last; ++e)
            {
                // printf("%f \n",*e);
                rfa.set_max_abs_val(*e);
                rfa.unsafe_add(*e);
                rfa.renorm();
            }
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
