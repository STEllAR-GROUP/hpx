//  Copyright (c) 2016 Minh-Khanh Do
//  Copyright (c) 2016-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/iterator_support/traits/is_iterator.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/algorithms/traits/segmented_iterator_traits.hpp>
#endif
#include <hpx/parallel/util/tagged_pair.hpp>

#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // transfer
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename FwdIter, typename OutIter>
        struct iterators_are_segmented
          : std::integral_constant<bool,
                hpx::traits::segmented_iterator_traits<
                    FwdIter>::is_segmented_iterator::value &&
                    hpx::traits::segmented_iterator_traits<
                        OutIter>::is_segmented_iterator::value>
        {
        };

        template <typename FwdIter, typename OutIter>
        struct iterators_are_not_segmented
          : std::integral_constant<bool,
                !hpx::traits::segmented_iterator_traits<
                    FwdIter>::is_segmented_iterator::value &&
                    !hpx::traits::segmented_iterator_traits<
                        OutIter>::is_segmented_iterator::value>
        {
        };

        ///////////////////////////////////////////////////////////////////////
        // parallel version
        template <typename Algo, typename ExPolicy, typename FwdIter1,
            typename Sent1, typename FwdIter2>
        typename util::detail::algorithm_result<ExPolicy,
            util::in_out_result<FwdIter1, FwdIter2>>::type
        transfer_(ExPolicy&& policy, FwdIter1 first, Sent1 last, FwdIter2 dest,
            std::false_type)
        {
            typedef hpx::is_sequenced_execution_policy<ExPolicy> is_seq;

            return Algo().call(
                std::forward<ExPolicy>(policy), is_seq(), first, last, dest);
        }

#if !defined(HPX_COMPUTE_DEVICE_CODE)
        // forward declare segmented version
        template <typename Algo, typename ExPolicy, typename FwdIter1,
            typename Sent1, typename FwdIter2>
        typename util::detail::algorithm_result<ExPolicy,
            util::in_out_result<FwdIter1, FwdIter2>>::type
        transfer_(ExPolicy&& policy, FwdIter1 first, Sent1 last, FwdIter2 dest,
            std::true_type);
#endif

        // Executes transfer algorithm on the elements in the range [first, last),
        // to another range beginning at \a dest.
        //
        // \note   Complexity: Performs exactly \a last - \a first transfer assignments.
        //
        //
        // \tparam Algo        The algorithm that is used to transfer the elements.
        //                     Should be hpx::parallel::detail::copy or
        //                     hpx::parallel::detail::move.
        // \tparam ExPolicy    The type of the execution policy to use (deduced).
        //                     It describes the manner in which the execution
        //                     of the algorithm may be parallelized and the manner
        //                     in which it executes the move assignments.
        // \tparam FwdIter1    The type of the source iterators used (deduced).
        //                     This iterator type must meet the requirements of an
        //                     forward iterator.
        // \tparam FwdIter2    The type of the iterator representing the
        //                     destination range (deduced).
        //                     This iterator type must meet the requirements of an
        //                     output iterator.
        //
        // \param policy       The execution policy to use for the scheduling of
        //                     the iterations.
        // \param first        Refers to the beginning of the sequence of elements
        //                     the algorithm will be applied to.
        // \param last         Refers to the end of the sequence of elements the
        //                     algorithm will be applied to.
        // \param dest         Refers to the beginning of the destination range.
        //
        // \returns  The \a transfer algorithm returns a \a hpx::future<FwdIter2> if
        //           the execution policy is of type
        //           \a sequenced_task_policy or
        //           \a parallel_task_policy and
        //           returns \a FwdIter2 otherwise.
        //           The \a move algorithm returns the output iterator to the
        //           element in the destination range, one past the last element
        //           transferred.
        //
        // clang-format off
        template <typename Algo, typename ExPolicy, typename FwdIter1,
            typename Sent1, typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_sentinel_for<Sent1, FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
        // clang-format on
        typename util::detail::algorithm_result<ExPolicy,
            util::in_out_result<FwdIter1, FwdIter2>>::type
        transfer(ExPolicy&& policy, FwdIter1 first, Sent1 last, FwdIter2 dest)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Required at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator<FwdIter2>::value ||
                    (hpx::is_sequenced_execution_policy<ExPolicy>::value &&
                        hpx::traits::is_output_iterator<FwdIter2>::value),
                "Requires at least forward iterator or sequential execution.");

#if defined(HPX_COMPUTE_DEVICE_CODE)
            return transfer_<Algo>(std::forward<ExPolicy>(policy), first, last,
                dest, std::false_type());
#else
            typedef hpx::traits::is_segmented_iterator<FwdIter1> is_segmented;

            return transfer_<Algo>(std::forward<ExPolicy>(policy), first, last,
                dest, is_segmented());
#endif
        }
    }    // namespace detail
}}}      // namespace hpx::parallel::v1
