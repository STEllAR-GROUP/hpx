//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/rotate.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_local/dataflow.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/pack_traversal/unwrap.hpp>
#include <hpx/parallel/util/tagged_pair.hpp>

#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/copy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/rotate.hpp>
#include <hpx/parallel/algorithms/reverse.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/result_types.hpp>
#include <hpx/parallel/util/transfer.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // rotate
    namespace detail {
        /// \cond NOINTERNAL
        template <typename ExPolicy, typename FwdIter>
        hpx::future<util::in_out_result<FwdIter, FwdIter>> rotate_helper(
            ExPolicy policy, FwdIter first, FwdIter new_first, FwdIter last)
        {
            typedef std::false_type non_seq;

            auto p = hpx::execution::parallel_task_policy()
                         .on(policy.executor())
                         .with(policy.parameters());

            detail::reverse<FwdIter> r;
            return dataflow(
                [=](hpx::future<FwdIter>&& f1,
                    hpx::future<FwdIter>&& f2) mutable
                -> hpx::future<util::in_out_result<FwdIter, FwdIter>> {
                    // propagate exceptions
                    f1.get();
                    f2.get();

                    hpx::future<FwdIter> f = r.call(p, non_seq(), first, last);
                    return f.then([=](hpx::future<FwdIter>&& f) mutable
                        -> util::in_out_result<FwdIter, FwdIter> {
                        f.get();    // propagate exceptions
                        std::advance(first, std::distance(new_first, last));
                        return util::in_out_result<FwdIter, FwdIter>{
                            first, last};
                    });
                },
                r.call(p, non_seq(), first, new_first),
                r.call(p, non_seq(), new_first, last));
        }

        template <typename IterPair>
        struct rotate : public detail::algorithm<rotate<IterPair>, IterPair>
        {
            rotate()
              : rotate::algorithm("rotate")
            {
            }

            template <typename ExPolicy, typename InIter>
            static IterPair sequential(
                ExPolicy, InIter first, InIter new_first, InIter last)
            {
                return detail::sequential_rotate(first, new_first, last);
            }

            template <typename ExPolicy, typename FwdIter>
            static typename util::detail::algorithm_result<ExPolicy,
                IterPair>::type
            parallel(ExPolicy&& policy, FwdIter first, FwdIter new_first,
                FwdIter last)
            {
                return util::detail::algorithm_result<ExPolicy, IterPair>::get(
                    rotate_helper(std::forward<ExPolicy>(policy), first,
                        new_first, last));
            }
        };
        /// \endcond
    }    // namespace detail

    /// Performs a left rotation on a range of elements. Specifically,
    /// \a rotate swaps the elements in the range [first, last) in such a way
    /// that the element new_first becomes the first element of the new range
    /// and new_first - 1 becomes the last element.
    ///
    /// \note   Complexity: Linear in the distance between \a first and \a last.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param new_first    Refers to the element that should appear at the
    ///                     beginning of the rotated range.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    ///
    /// The assignments in the parallel \a rotate algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a rotate algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \note The type of dereferenced \a FwdIter must meet the requirements
    ///       of \a MoveAssignable and \a MoveConstructible.
    ///
    /// \returns  The \a rotate algorithm returns a
    ///           \a hpx::future<tagged_pair<tag::begin(FwdIter), tag::end(FwdIter)> >
    ///           if the execution policy is of type
    ///           \a parallel_task_policy and
    ///           returns \a tagged_pair<tag::begin(FwdIter), tag::end(FwdIter)>
    ///           otherwise.
    ///           The \a rotate algorithm returns the iterator equal to
    ///           pair(first + (last - new_first), last).
    ///
    template <typename ExPolicy, typename FwdIter,
        HPX_CONCEPT_REQUIRES_(hpx::is_execution_policy<ExPolicy>::value&&
                hpx::traits::is_iterator<FwdIter>::value)>
    typename util::detail::algorithm_result<ExPolicy,
        util::in_out_result<FwdIter, FwdIter>>::type
    rotate(ExPolicy&& policy, FwdIter first, FwdIter new_first, FwdIter last)
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
            "Requires at least forward iterator.");

        typedef std::integral_constant<bool,
            hpx::is_sequenced_execution_policy<ExPolicy>::value ||
                !hpx::traits::is_bidirectional_iterator<FwdIter>::value>
            is_seq;

        return detail::rotate<util::in_out_result<FwdIter, FwdIter>>().call(
            std::forward<ExPolicy>(policy), is_seq(), first, new_first, last);
    }

    ///////////////////////////////////////////////////////////////////////////
    // rotate_copy
    namespace detail {
        /// \cond NOINTERNAL

        // sequential rotate_copy
        template <typename InIter, typename OutIter>
        inline util::in_out_result<InIter, OutIter> sequential_rotate_copy(
            InIter first, InIter new_first, InIter last, OutIter dest_first)
        {
            util::in_out_result<InIter, OutIter> p1 =
                util::copy(new_first, last, dest_first);
            util::in_out_result<InIter, OutIter> p2 =
                util::copy(first, new_first, std::move(p1.out));
            return util::in_out_result<InIter, OutIter>{
                std::move(p1.in), std::move(p2.out)};
        }

        template <typename ExPolicy, typename FwdIter1, typename FwdIter2>
        hpx::future<util::in_out_result<FwdIter1, FwdIter2>> rotate_copy_helper(
            ExPolicy policy, FwdIter1 first, FwdIter1 new_first, FwdIter1 last,
            FwdIter2 dest_first)
        {
            typedef std::false_type non_seq;

            auto p = hpx::execution::parallel_task_policy()
                         .on(policy.executor())
                         .with(policy.parameters());

            typedef util::in_out_result<FwdIter1, FwdIter2> copy_return_type;

            hpx::future<copy_return_type> f =
                detail::copy<copy_return_type>().call(
                    p, non_seq(), new_first, last, dest_first);

            return f.then([=](hpx::future<copy_return_type>&& result)
                              -> hpx::future<copy_return_type> {
                copy_return_type p1 = result.get();
                return detail::copy<copy_return_type>().call(
                    p, non_seq(), first, new_first, p1.out);
            });
        }

        template <typename IterPair>
        struct rotate_copy
          : public detail::algorithm<rotate_copy<IterPair>, IterPair>
        {
            rotate_copy()
              : rotate_copy::algorithm("rotate_copy")
            {
            }

            template <typename ExPolicy, typename InIter, typename OutIter>
            static util::in_out_result<InIter, OutIter> sequential(ExPolicy,
                InIter first, InIter new_first, InIter last, OutIter dest_first)
            {
                return sequential_rotate_copy(
                    first, new_first, last, dest_first);
            }

            template <typename ExPolicy, typename FwdIter1, typename FwdIter2>
            static typename util::detail::algorithm_result<ExPolicy,
                util::in_out_result<FwdIter1, FwdIter2>>::type
            parallel(ExPolicy&& policy, FwdIter1 first, FwdIter1 new_first,
                FwdIter1 last, FwdIter2 dest_first)
            {
                return util::detail::algorithm_result<ExPolicy, IterPair>::get(
                    rotate_copy_helper(std::forward<ExPolicy>(policy), first,
                        new_first, last, dest_first));
            }
        };
        /// \endcond
    }    // namespace detail

    /// Copies the elements from the range [first, last), to another range
    /// beginning at \a dest_first in such a way, that the element
    /// \a new_first becomes the first element of the new range and
    /// \a new_first - 1 becomes the last element.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     bidirectional iterator.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param new_first    Refers to the element that should appear at the
    ///                     beginning of the rotated range.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest_first   Refers to the begin of the destination range.
    ///
    /// The assignments in the parallel \a rotate_copy algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a rotate_copy algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a rotate_copy algorithm returns a
    ///           \a hpx::future<tagged_pair<tag::in(FwdIter1), tag::out(FwdIter2)> >
    ///           if the execution policy is of type
    ///           \a parallel_task_policy and
    ///           returns \a tagged_pair<tag::in(FwdIter1), tag::out(FwdIter2)>
    ///           otherwise.
    ///           The \a rotate_copy algorithm returns the output iterator to the
    ///           element past the last element copied.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        HPX_CONCEPT_REQUIRES_(hpx::traits::is_iterator<FwdIter1>::value&&
                hpx::is_execution_policy<ExPolicy>::value&&
                    hpx::traits::is_iterator<FwdIter2>::value)>
    typename util::detail::algorithm_result<ExPolicy,
        util::in_out_result<FwdIter1, FwdIter2>>::type
    rotate_copy(ExPolicy&& policy, FwdIter1 first, FwdIter1 new_first,
        FwdIter1 last, FwdIter2 dest_first)
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Requires at least forward iterator.");
        static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");

        typedef std::integral_constant<bool,
            hpx::is_sequenced_execution_policy<ExPolicy>::value ||
                !hpx::traits::is_bidirectional_iterator<FwdIter1>::value>
            is_seq;

        return detail::rotate_copy<util::in_out_result<FwdIter1, FwdIter2>>()
            .call(std::forward<ExPolicy>(policy), is_seq(), first, new_first,
                last, dest_first);
    }
}}}    // namespace hpx::parallel::v1
