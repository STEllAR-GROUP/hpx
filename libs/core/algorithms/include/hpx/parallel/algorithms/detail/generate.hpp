//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/parallel/util/loop.hpp>

#include <algorithm>
#include <cstddef>
#include <utility>

namespace hpx::parallel::detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename Sent, typename F>
    constexpr Iter sequential_generate_helper(Iter first, Sent last, F&& f)
    {
        return util::loop_ind<hpx::execution::sequenced_policy>(
            first, last, [f = HPX_FORWARD(F, f)](auto& v) mutable { v = f(); });
    }

    struct sequential_generate_t
      : hpx::functional::detail::tag_fallback<sequential_generate_t>
    {
    private:
        template <typename ExPolicy, typename Iter, typename Sent, typename F>
        friend constexpr Iter tag_fallback_invoke(
            sequential_generate_t, ExPolicy&&, Iter first, Sent last, F&& f)
        {
            return sequential_generate_helper(first, last, HPX_FORWARD(F, f));
        }
    };

#if !defined(HPX_COMPUTE_DEVICE_CODE)
    inline constexpr sequential_generate_t sequential_generate =
        sequential_generate_t{};
#else
    template <typename ExPolicy, typename Iter, typename Sent, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Iter sequential_generate(
        ExPolicy&& policy, Iter first, Sent last, F&& f)
    {
        return sequential_generate_t{}(
            HPX_FORWARD(ExPolicy, policy), first, last, HPX_FORWARD(F, f));
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename F>
    constexpr Iter sequential_generate_n_helper(
        Iter first, std::size_t count, F&& f)
    {
        return std::generate_n(first, count, f);
    }

    struct sequential_generate_n_t
      : hpx::functional::detail::tag_fallback<sequential_generate_n_t>
    {
    private:
        template <typename ExPolicy, typename Iter, typename F>
        friend constexpr Iter tag_fallback_invoke(sequential_generate_n_t,
            ExPolicy&&, Iter first, std::size_t count, F&& f)
        {
            return sequential_generate_n_helper(
                first, count, HPX_FORWARD(F, f));
        }
    };

#if !defined(HPX_COMPUTE_DEVICE_CODE)
    inline constexpr sequential_generate_n_t sequential_generate_n =
        sequential_generate_n_t{};
#else
    template <typename ExPolicy, typename Iter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Iter sequential_generate_n(
        ExPolicy&& policy, Iter first, std::size_t count, F&& f)
    {
        return sequential_generate_n_t{}(
            HPX_FORWARD(ExPolicy, policy), first, count, HPX_FORWARD(F, f));
    }
#endif

}    // namespace hpx::parallel::detail
