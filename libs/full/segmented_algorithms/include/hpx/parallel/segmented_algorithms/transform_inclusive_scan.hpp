//  Copyright (c) 2017 Ajai V George
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/segmented_algorithms/transform_inclusive_scan.hpp

#pragma once

#include <hpx/config.hpp>

#include <hpx/algorithms/traits/segmented_iterator_traits.hpp>

#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/transform_inclusive_scan.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/segmented_algorithms/detail/scan.hpp>
#include <hpx/parallel/segmented_algorithms/inclusive_scan.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>

#include <type_traits>
#include <utility>
#include <vector>

// The segmented iterators we support all live in namespace hpx::segmented
namespace hpx { namespace segmented {
    // clang-format off
    template <typename InIter, typename OutIter, typename Op,
        typename Conv,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_iterator<InIter>::value &&
            hpx::traits::is_segmented_iterator<InIter>::value &&
            hpx::traits::is_iterator<OutIter>::value &&
            hpx::traits::is_segmented_iterator<OutIter>::value
        )>
    // clang-format on
    OutIter tag_invoke(hpx::transform_inclusive_scan_t, InIter first,
        InIter last, OutIter dest, Op&& op, Conv&& conv)
    {
        static_assert(hpx::traits::is_input_iterator<InIter>::value,
            "Requires at least input iterator.");

        static_assert(hpx::traits::is_output_iterator<OutIter>::value,
            "Requires at least output iterator.");

        if (first == last)
            return dest;

        using value_type = typename std::iterator_traits<InIter>::value_type;

        return hpx::parallel::detail::segmented_inclusive_scan(
            hpx::execution::seq, first, last, dest, value_type{},
            HPX_FORWARD(Op, op), std::true_type{}, HPX_FORWARD(Conv, conv));
    }

    // clang-format off
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename Op, typename Conv,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<FwdIter1>::value &&
            hpx::traits::is_segmented_iterator<FwdIter1>::value &&
            hpx::traits::is_iterator<FwdIter2>::value &&
            hpx::traits::is_segmented_iterator<FwdIter2>::value
        )>
    // clang-format on
    typename parallel::util::detail::algorithm_result<ExPolicy, FwdIter2>::type
    tag_invoke(hpx::transform_inclusive_scan_t, ExPolicy&& policy,
        FwdIter1 first, FwdIter1 last, FwdIter2 dest, Op&& op, Conv&& conv)
    {
        static_assert(hpx::traits::is_forward_iterator<FwdIter1>::value,
            "Requires at least forward iterator.");

        static_assert(hpx::traits::is_forward_iterator<FwdIter2>::value,
            "Requires at least forward iterator.");

        if (first == last)
            return parallel::util::detail::algorithm_result<ExPolicy,
                FwdIter2>::get(HPX_MOVE(dest));

        using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;
        using value_type = typename std::iterator_traits<FwdIter1>::value_type;

        return hpx::parallel::detail::segmented_inclusive_scan(
            HPX_FORWARD(ExPolicy, policy), first, last, dest, value_type{},
            HPX_FORWARD(Op, op), is_seq(), HPX_FORWARD(Conv, conv));
    }

    // clang-format off
    template <typename InIter, typename OutIter,
        typename T, typename Op, typename Conv,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_iterator<InIter>::value &&
            hpx::traits::is_segmented_iterator<InIter>::value &&
            hpx::traits::is_iterator<OutIter>::value &&
            hpx::traits::is_segmented_iterator<OutIter>::value
        )>
    // clang-format on
    OutIter tag_invoke(hpx::transform_inclusive_scan_t, InIter first,
        InIter last, OutIter dest, Op&& op, Conv&& conv, T init)
    {
        static_assert(hpx::traits::is_input_iterator<InIter>::value,
            "Requires at least input iterator.");

        static_assert(hpx::traits::is_output_iterator<OutIter>::value,
            "Requires at least output iterator.");

        if (first == last)
            return dest;

        return hpx::parallel::detail::segmented_inclusive_scan(
            hpx::execution::seq, first, last, dest, HPX_MOVE(init),
            HPX_FORWARD(Op, op), std::true_type{}, HPX_FORWARD(Conv, conv));
    }

    // clang-format off
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename T, typename Op, typename Conv,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<FwdIter1>::value &&
            hpx::traits::is_segmented_iterator<FwdIter1>::value &&
            hpx::traits::is_iterator<FwdIter2>::value &&
            hpx::traits::is_segmented_iterator<FwdIter2>::value
        )>
    // clang-format on
    typename parallel::util::detail::algorithm_result<ExPolicy, FwdIter2>::type
    tag_invoke(hpx::transform_inclusive_scan_t, ExPolicy&& policy,
        FwdIter1 first, FwdIter1 last, FwdIter2 dest, Op&& op, Conv&& conv,
        T init)
    {
        static_assert(hpx::traits::is_forward_iterator<FwdIter1>::value,
            "Requires at least forward iterator.");

        static_assert(hpx::traits::is_forward_iterator<FwdIter2>::value,
            "Requires at least forward iterator.");

        if (first == last)
            return parallel::util::detail::algorithm_result<ExPolicy,
                FwdIter2>::get(HPX_MOVE(dest));

        using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;

        return hpx::parallel::detail::segmented_inclusive_scan(
            HPX_FORWARD(ExPolicy, policy), first, last, dest, HPX_MOVE(init),
            HPX_FORWARD(Op, op), is_seq(), HPX_FORWARD(Conv, conv));
    }
}}    // namespace hpx::segmented
