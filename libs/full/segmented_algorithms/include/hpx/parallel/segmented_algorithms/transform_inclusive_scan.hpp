//  Copyright (c) 2017 Ajai V George
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
#include <hpx/parallel/util/projection_identity.hpp>

#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // segmented transform_inclusive_scan
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ExPolicy, typename InIter, typename OutIter,
            typename Op, typename Conv, typename T>
        typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        transform_inclusive_scan_(ExPolicy&& policy, InIter first, InIter last,
            OutIter dest, Conv&& conv, T&& init, Op&& op, std::true_type)
        {
            if (first == last)
            {
                return util::detail::algorithm_result<ExPolicy, OutIter>::get(
                    std::move(dest));
            }

            using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;

            return hpx::parallel::v1::detail::segmented_inclusive_scan(
                std::forward<ExPolicy>(policy), first, last, dest,
                std::forward<T>(init), std::forward<Op>(op), is_seq(),
                std::forward<Conv>(conv));
        }

        template <typename ExPolicy, typename InIter, typename OutIter,
            typename Op, typename Conv>
        typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        transform_inclusive_scan_(ExPolicy&& policy, InIter first, InIter last,
            OutIter dest, Conv&& conv, Op&& op, std::true_type)
        {
            if (first == last)
            {
                return util::detail::algorithm_result<ExPolicy, OutIter>::get(
                    std::move(dest));
            }

            using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;
            using value_type =
                typename std::iterator_traits<InIter>::value_type;

            return hpx::parallel::v1::detail::segmented_inclusive_scan(
                std::forward<ExPolicy>(policy), first, last, dest, value_type{},
                std::forward<Op>(op), is_seq(), std::forward<Conv>(conv));
        }

        // forward declare the non-segmented version of this algorithm
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Op, typename Conv, typename T>
        typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type
        transform_inclusive_scan_(ExPolicy&& policy, FwdIter1 first,
            FwdIter1 last, FwdIter2 dest, Conv&& conv, T&& init, Op&& op,
            std::false_type);

        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Op, typename Conv>
        typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type
        transform_inclusive_scan_(ExPolicy&& policy, FwdIter1 first,
            FwdIter1 last, FwdIter2 dest, Conv&& conv, Op&& op,
            std::false_type);

        /// \endcond
    }    // namespace detail
}}}      // namespace hpx::parallel::v1
