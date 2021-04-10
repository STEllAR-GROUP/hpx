
//  Copyright (c) 2021 @rainmaker6
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/shift_left.hpp
//  Copyright (c) 2021 @rainmaker6
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/shift_left.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_local/dataflow.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/execution.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
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
    // shift_left
    namespace detail {
        /// \cond NOINTERNAL
        template <typename ExPolicy, typename FwdIter>
        hpx::future<util::in_out_result<FwdIter, FwdIter>> shift_left_helper(
            ExPolicy policy, FwdIter first, FwdIter new_first, FwdIter last)
        {
            using non_seq = std::false_type;

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

                    hpx::future<FwdIter> f = r.call2(p, non_seq(), first, last);
                    return f.then([=](hpx::future<FwdIter>&& f) mutable
                        -> util::in_out_result<FwdIter, FwdIter> {
                        f.get();    // propagate exceptions
                        std::advance(first, std::distance(new_first, last));
                        return util::in_out_result<FwdIter, FwdIter>{
                            first, last};
                    });
                },
                r.call2(p, non_seq(), first, new_first),
                r.call2(p, non_seq(), new_first, last));
        }

        template <typename IterPair>
        struct shift_left
          : public detail::algorithm<shift_left<IterPair>, IterPair>
        {
            shift_left()
              : shift_left::algorithm("shift_left")
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
                    shift_left_helper(std::forward<ExPolicy>(policy), first,
                        new_first, last));
            }
        };
        /// \endcond
    }    // namespace detail

    /// Performs a left rotation on a range of elements. Specifically,
    /// \a shift_left swaps the elements in the range [first, last) in such a way
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
    /// The assignments in the parallel \a shift_left algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a shift_left algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \note The type of dereferenced \a FwdIter must meet the requirements
    ///       of \a MoveAssignable and \a MoveConstructible.
    ///
    /// \returns  The \a shift_left algorithm returns a
    ///           \a hpx::future<tagged_pair<tag::begin(FwdIter), tag::end(FwdIter)>
    ///           > if the execution policy is of type \a parallel_task_policy and
    ///           returns \a tagged_pair<tag::begin(FwdIter), tag::end(FwdIter)>
    ///           otherwise.
    ///           The \a shift_left algorithm returns the iterator equal to
    ///           pair(first + (last - new_first), last).
    ///

}}}    // namespace hpx::parallel::v1
