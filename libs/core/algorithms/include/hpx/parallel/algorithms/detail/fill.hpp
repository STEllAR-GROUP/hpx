//  Copyright (c) 2020 Hartmut Kaiser
//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>

#include <algorithm>
#include <cstddef>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename Sent, typename T>
    constexpr Iter sequential_fill_helper(Iter first, Sent last, T const& value)
    {
        for (; first != last; ++first)
        {
            *first = value;
        }
        return first;
    }

    struct sequential_fill_t : hpx::functional::tag_fallback<sequential_fill_t>
    {
    private:
        template <typename ExPolicy, typename Iter, typename Sent, typename T>
        friend constexpr Iter tag_fallback_dispatch(sequential_fill_t,
            ExPolicy&&, Iter first, Sent last, T const& value)
        {
            return sequential_fill_helper(first, last, value);
        }
    };

#if !defined(HPX_COMPUTE_DEVICE_CODE)
    HPX_INLINE_CONSTEXPR_VARIABLE sequential_fill_t sequential_fill =
        sequential_fill_t{};
#else
    template <typename ExPolicy, typename Iter, typename Sent, typename T>
    HPX_HOST_DEVICE HPX_FORCEINLINE Iter sequential_fill(
        ExPolicy&& policy, Iter first, Sent last, T const& value)
    {
        return sequential_fill_t{}(
            std::forward<ExPolicy>(policy), first, last, value);
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename T>
    constexpr Iter sequential_fill_n_helper(
        Iter first, std::size_t count, T const& value)
    {
        return std::fill_n(first, count, value);
    }

    struct sequential_fill_n_t
      : hpx::functional::tag_fallback<sequential_fill_n_t>
    {
    private:
        template <typename ExPolicy, typename Iter, typename T>
        friend constexpr Iter tag_fallback_dispatch(sequential_fill_n_t,
            ExPolicy&&, Iter first, std::size_t count, T const& value)
        {
            return sequential_fill_n_helper(first, count, value);
        }
    };

#if !defined(HPX_COMPUTE_DEVICE_CODE)
    HPX_INLINE_CONSTEXPR_VARIABLE sequential_fill_n_t sequential_fill_n =
        sequential_fill_n_t{};
#else
    template <typename ExPolicy, typename Iter, typename T>
    HPX_HOST_DEVICE HPX_FORCEINLINE Iter sequential_fill_n(
        ExPolicy&& policy, Iter first, std::size_t count, T const& value)
    {
        return sequential_fill_n_t{}(
            std::forward<ExPolicy>(policy), first, count, value);
    }
#endif

}}}}    // namespace hpx::parallel::v1::detail
