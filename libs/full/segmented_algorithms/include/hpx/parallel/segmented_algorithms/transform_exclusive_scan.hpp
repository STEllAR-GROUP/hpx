//  Copyright (c) 2017 Ajai V George
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/segmented_algorithms/transform_exclusive_scan.hpp

#pragma once

#include <hpx/config.hpp>

#include <hpx/algorithms/traits/segmented_iterator_traits.hpp>

#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/transform_exclusive_scan.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/segmented_algorithms/detail/scan.hpp>
#include <hpx/parallel/segmented_algorithms/exclusive_scan.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <type_traits>
#include <utility>
#include <vector>

// The segmented iterators we support all live in namespace hpx::segmented
namespace hpx { namespace segmented {
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
    OutIter tag_dispatch(hpx::transform_exclusive_scan_t, InIter first,
        InIter last, OutIter dest, T init, Op&& op, Conv&& conv)
    {
        static_assert(hpx::traits::is_input_iterator<InIter>::value,
            "Requires at least input iterator.");

        static_assert(hpx::traits::is_output_iterator<OutIter>::value,
            "Requires at least output iterator.");

        if (first == last)
            return dest;

        return hpx::parallel::v1::detail::segmented_exclusive_scan(
            hpx::execution::seq, first, last, dest, std::move(init),
            std::forward<Op>(op), std::true_type{}, std::forward<Conv>(conv));
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
    tag_dispatch(hpx::transform_exclusive_scan_t, ExPolicy&& policy,
        FwdIter1 first, FwdIter1 last, FwdIter2 dest, T init, Op&& op,
        Conv&& conv)
    {
        static_assert(hpx::traits::is_forward_iterator<FwdIter1>::value,
            "Requires at least forward iterator.");

        static_assert(hpx::traits::is_forward_iterator<FwdIter2>::value,
            "Requires at least forward iterator.");

        if (first == last)
            return parallel::util::detail::algorithm_result<ExPolicy,
                FwdIter2>::get(std::move(dest));

        using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;

        return hpx::parallel::v1::detail::segmented_exclusive_scan(
            std::forward<ExPolicy>(policy), first, last, dest, std::move(init),
            std::forward<Op>(op), is_seq(), std::forward<Conv>(conv));
    }
}}    // namespace hpx::segmented
