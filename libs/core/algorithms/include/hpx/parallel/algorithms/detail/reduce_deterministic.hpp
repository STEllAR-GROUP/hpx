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
#include <hpx/type_support/pack.hpp>

#include <cstddef>
#include <cstring>
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
            /// TODO: Put constraint on Reduce to be a binary plus operator
            (void) r;

            // hpx_rfa_bin_host_buffer should be initialized by the frontend of
            // this method

            hpx::parallel::detail::rfa::reproducible_floating_accumulator<T>
                rfa;
            rfa.set_max_abs_val(init);
            rfa.unsafe_add(init);
            rfa.renorm();
            size_t count = 0;
            T max_val = static_cast<T>(0.0);
            for (auto e = first; e != last; ++e)
            {
                T temp_max_val = std::abs(static_cast<T>(*e));
                if (max_val < temp_max_val)
                {
                    rfa.set_max_abs_val(temp_max_val);
                    max_val = temp_max_val;
                }
                rfa.unsafe_add(*e);
                count++;
                if (count == rfa.endurance())
                {
                    rfa.renorm();
                    count = 0;
                }
            }
            return rfa.conv();
        }
    };

    template <typename ExPolicy>
    struct sequential_reduce_deterministic_rfa_t final
      : hpx::functional::detail::tag_fallback<
            sequential_reduce_deterministic_rfa_t<ExPolicy>>
    {
    private:
        template <typename InIterB, typename T>
        friend constexpr hpx::parallel::detail::rfa::
            reproducible_floating_accumulator<T>
            tag_fallback_invoke(sequential_reduce_deterministic_rfa_t,
                ExPolicy&&, InIterB first, std::size_t partition_size, T init,
                std::true_type&&)
        {
            // hpx_rfa_bin_host_buffer should be initialized by the frontend of
            // this method

            hpx::parallel::detail::rfa::reproducible_floating_accumulator<T>
                rfa;
            rfa.zero();
            rfa += init;
            size_t count = 0;
            T max_val = static_cast<T>(0.0);
            std::size_t partition_size_lim = 0;
            for (auto e = first; partition_size_lim < partition_size;
                ++partition_size_lim, ++e)
            {
                T temp_max_val = std::abs(static_cast<T>(*e));
                if (max_val < temp_max_val)
                {
                    rfa.set_max_abs_val(temp_max_val);
                    max_val = temp_max_val;
                }
                rfa.unsafe_add(*e);
                count++;
                if (count == rfa.endurance())
                {
                    rfa.renorm();
                    count = 0;
                }
            }
            return rfa;
        }

        template <typename InIterB, typename T>
        friend constexpr T tag_fallback_invoke(
            sequential_reduce_deterministic_rfa_t, ExPolicy&&, InIterB first,
            std::size_t partition_size, T init, std::false_type&&)
        {
            // hpx_rfa_bin_host_buffer should be initialized by the frontend of
            // this method

            T rfa;
            rfa.zero();
            rfa += init;
            std::size_t partition_size_lim = 0;
            for (auto e = first; partition_size_lim < partition_size;
                ++partition_size_lim, ++e)
            {
                rfa += (*e);
            }
            return rfa;
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

#if !defined(HPX_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    inline constexpr sequential_reduce_deterministic_rfa_t<ExPolicy>
        sequential_reduce_deterministic_rfa =
            sequential_reduce_deterministic_rfa_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename... Args>
    HPX_HOST_DEVICE HPX_FORCEINLINE auto sequential_reduce_deterministic_rfa(
        Args&&... args)
    {
        return sequential_reduce_deterministic_rfa_t<ExPolicy>{}(
            std::forward<Args>(args)...);
    }
#endif
}    // namespace hpx::parallel::detail
