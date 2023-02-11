//  Copyright (c) 2022 A Kishore Kumar
//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/reduce.hpp>
#include <hpx/parallel/unseq/reduce_helpers.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx::parallel::detail {

    ///////////////////////////////////////////////////////////////////////////
    // clang-format off
    template <typename ExPolicy, typename InIterB, typename InIterE, typename T,
        typename Reduce,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_unsequenced_execution_policy_v<ExPolicy>
        )>
    // clang-format on
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr T tag_invoke(
        sequential_reduce_t<ExPolicy>, ExPolicy&&, InIterB first, InIterE last,
        T init, Reduce&& r)
    {
        if constexpr (hpx::traits::is_random_access_iterator_v<InIterB>)
        {
            return hpx::parallel::util::detail::unseq_reduce_n::reduce(first,
                std::distance(first, last), HPX_FORWARD(T, init),
                HPX_FORWARD(Reduce, r), [&](auto const v) { return v; });
        }
        else
        {
            using sequenced_policy = hpx::execution::sequenced_policy;
            return sequential_reduce<sequenced_policy>(
                sequenced_policy{}, first, last, init, HPX_FORWARD(Reduce, r));
        }
    }

    // clang-format off
    template <typename ExPolicy, typename T, typename FwdIter, typename Reduce,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_unsequenced_execution_policy_v<ExPolicy>
        )>
    // clang-format on
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr T tag_invoke(
        sequential_reduce_t<ExPolicy>, FwdIter part_begin,
        std::size_t part_size, T init, Reduce r)
    {
        if constexpr (hpx::traits::is_random_access_iterator_v<FwdIter>)
        {
            return hpx::parallel::util::detail::unseq_reduce_n::reduce(
                part_begin, part_size, HPX_FORWARD(T, init),
                HPX_FORWARD(Reduce, r), [&](auto const v) { return v; });
        }
        else
        {
            return sequential_reduce<hpx::execution::sequenced_policy>(
                part_begin, part_size, init, r);
        }
    }

    // clang-format off
    template <typename ExPolicy, typename Iter, typename Sent, typename T,
        typename Reduce, typename Convert,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_unsequenced_execution_policy_v<ExPolicy>
        )>
    // clang-format on
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr T tag_invoke(
        sequential_reduce_t<ExPolicy>, ExPolicy&&, Iter first, Sent last,
        T init, Reduce&& r, Convert&& conv)
    {
        if constexpr (hpx::traits::is_random_access_iterator_v<Iter>)
        {
            return hpx::parallel::util::detail::unseq_reduce_n::reduce(first,
                std::distance(first, last), HPX_FORWARD(T, init),
                HPX_FORWARD(Reduce, r), HPX_FORWARD(Convert, conv));
        }
        else
        {
            using sequenced_policy = hpx::execution::sequenced_policy;
            return sequential_reduce<sequenced_policy>(sequenced_policy{},
                first, last, init, HPX_FORWARD(Reduce, r),
                HPX_FORWARD(Convert, conv));
        }
    }

    // clang-format off
    template <typename ExPolicy, typename T, typename Iter, typename Reduce,
        typename Convert,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_unsequenced_execution_policy_v<ExPolicy>
        )>
    // clang-format on
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr T tag_invoke(
        sequential_reduce_t<ExPolicy>, Iter part_begin, std::size_t part_size,
        T init, Reduce r, Convert conv)
    {
        if constexpr (hpx::traits::is_random_access_iterator_v<Iter>)
        {
            return hpx::parallel::util::detail::unseq_reduce_n::reduce(
                part_begin, part_size, HPX_FORWARD(T, init),
                HPX_FORWARD(Reduce, r), HPX_FORWARD(Convert, conv));
        }
        else
        {
            return sequential_reduce<hpx::execution::sequenced_policy>(
                part_begin, part_size, init, r, conv);
        }
    }

    // clang-format off
    template <typename ExPolicy, typename Iter1, typename Sent, typename Iter2,
        typename T, typename Reduce, typename Convert,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_unsequenced_execution_policy_v<ExPolicy>
        )>
    // clang-format on
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr T tag_invoke(
        sequential_reduce_t<ExPolicy>, Iter1 first1, Sent last1, Iter2 first2,
        T init, Reduce&& r, Convert&& conv)
    {
        constexpr bool iterators_are_random_access =
            hpx::traits::is_random_access_iterator_v<Iter1> &&
            hpx::traits::is_random_access_iterator_v<Iter2>;

        if constexpr (iterators_are_random_access)
        {
            return hpx::parallel::util::detail::unseq_binary_reduce_n::reduce(
                first1, first2, std::distance(first1, last1),
                HPX_FORWARD(T, init), HPX_FORWARD(Reduce, r),
                HPX_FORWARD(Convert, conv));
        }
        else
        {
            return sequential_reduce<hpx::execution::sequenced_policy>(first1,
                last1, first2, init, HPX_FORWARD(Reduce, r),
                HPX_FORWARD(Convert, conv));
        }
    }
}    // namespace hpx::parallel::detail
