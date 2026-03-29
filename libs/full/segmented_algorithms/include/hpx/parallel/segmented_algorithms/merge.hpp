//  Copyright (c) 2026 Abhishek Bansal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <hpx/modules/algorithms.hpp>
#include <hpx/modules/executors.hpp>
#include <hpx/parallel/segmented_algorithms/detail/merge.hpp>

#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx::segmented {

    // clang-format off
    template <typename InIter1, typename InIter2, typename DestIter,
        typename Comp = hpx::parallel::detail::less>
        requires(hpx::traits::is_iterator_v<InIter1> &&
            hpx::traits::is_segmented_iterator_v<InIter1> &&
            hpx::traits::is_iterator_v<InIter2> &&
            hpx::traits::is_segmented_iterator_v<InIter2> &&
            hpx::traits::is_iterator_v<DestIter> &&
            hpx::traits::is_segmented_iterator_v<DestIter> &&
            hpx::is_invocable_v<Comp,
                typename std::iterator_traits<InIter1>::value_type,
                typename std::iterator_traits<InIter2>::value_type>)
    DestIter tag_invoke(hpx::merge_t, InIter1 first1, InIter1 last1,
        InIter2 first2, InIter2 last2, DestIter dest, Comp comp = Comp())
    // clang-format on
    {
        return hpx::parallel::detail::segmented_merge(hpx::execution::seq,
            first1, last1, first2, last2, dest, HPX_MOVE(comp));
    }

    // clang-format off
    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename DestIter, typename Comp = hpx::parallel::detail::less>
        requires(hpx::is_execution_policy_v<ExPolicy> &&
            hpx::traits::is_iterator_v<InIter1> &&
            hpx::traits::is_segmented_iterator_v<InIter1> &&
            hpx::traits::is_iterator_v<InIter2> &&
            hpx::traits::is_segmented_iterator_v<InIter2> &&
            hpx::traits::is_iterator_v<DestIter> &&
            hpx::traits::is_segmented_iterator_v<DestIter> &&
            hpx::is_invocable_v<Comp,
                typename std::iterator_traits<InIter1>::value_type,
                typename std::iterator_traits<InIter2>::value_type>)
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, DestIter>
    tag_invoke(hpx::merge_t, ExPolicy&& policy, InIter1 first1,
        InIter1 last1, InIter2 first2, InIter2 last2, DestIter dest,
        Comp comp = Comp())
    // clang-format on
    {
        return hpx::parallel::detail::segmented_merge(
            HPX_FORWARD(ExPolicy, policy), first1, last1, first2, last2, dest,
            HPX_MOVE(comp));
    }
}    // namespace hpx::segmented
