//  Copyright (c) 2017 Ajai V George
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/segmented_algorithms/transform_inclusive_scan.hpp

#if !defined(HPX_PARALLEL_SEGMENTED_ALGORITHMS_TRANSFORM_INCLUSIVE_SCAN)
#define HPX_PARALLEL_SEGMENTED_ALGORITHMS_TRANSFORM_INCLUSIVE_SCAN

#include <hpx/config.hpp>

#include <hpx/traits/segmented_iterator_traits.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/transform_inclusive_scan.hpp>
#include <hpx/parallel/segmented_algorithms/inclusive_scan.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/segmented_algorithms/detail/scan.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1
{
    ///////////////////////////////////////////////////////////////////////////
    // segmented transform_inclusive_scan
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ExPolicy, typename InIter, typename OutIter,
            typename Op, typename Conv, typename T>
        typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        transform_inclusive_scan_(ExPolicy && policy, InIter first, InIter last,
            OutIter dest, Conv && conv, T init, Op && op, std::true_type)
        {
            if (first == last)
                return util::detail::algorithm_result<
                    ExPolicy, OutIter>::get(std::move(dest));

            return inclusive_scan_(std::forward<ExPolicy>(policy), first, last,
                dest, init, std::forward<Op>(op), std::true_type(),
                std::forward<Conv>(conv));
        }

        // forward declare the non-segmented version of this algorithm
        template <typename ExPolicy, typename InIter, typename OutIter,
            typename Op, typename Conv, typename T>
        typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        transform_inclusive_scan_(ExPolicy && policy, InIter first, InIter last,
            OutIter dest, Conv && conv, T init, Op && op, std::false_type);

        /// \endcond
    }
}}}
#endif
