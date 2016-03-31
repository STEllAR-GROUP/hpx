//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/rotate.hpp

#if !defined(HPX_PARALLEL_DETAIL_ROTATE_AUG_05_2014_0138PM)
#define HPX_PARALLEL_DETAIL_ROTATE_AUG_05_2014_0138PM

#include <hpx/config.hpp>
#include <hpx/traits/concepts.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/unwrapped.hpp>
#include <hpx/util/tagged_pair.hpp>
#include <hpx/dataflow.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/reverse.hpp>
#include <hpx/parallel/algorithms/copy.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // rotate
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename FwdIter>
        void sequential_rotate_helper(FwdIter first, FwdIter new_first,
            FwdIter last)
        {
            FwdIter next = new_first;
            while (first != next)
            {
                std::iter_swap(first++, next++);
                if (next == last)
                {
                    next = new_first;
                }
                else if (first == new_first)
                {
                    new_first = next;
                }
            }
        }

        template <typename FwdIter>
        inline std::pair<FwdIter, FwdIter>
        sequential_rotate(FwdIter first, FwdIter new_first, FwdIter last)
        {
            if (first != new_first && new_first != last)
                sequential_rotate_helper(first, new_first, last);

            std::advance(first, std::distance(new_first, last));
            return std::make_pair(first, last);
        }

        template <typename ExPolicy, typename FwdIter>
        hpx::future<std::pair<FwdIter, FwdIter> >
        rotate_helper(ExPolicy policy, FwdIter first, FwdIter new_first,
            FwdIter last)
        {
            typedef std::false_type non_seq;

            parallel_task_execution_policy p =
                parallel_task_execution_policy()
                    .on(policy.executor())
                    .with(policy.parameters());

            detail::reverse<FwdIter> r;
            return dataflow(
                [=](hpx::future<FwdIter>&& f1, hpx::future<FwdIter>&& f2) mutable
                ->  hpx::future<std::pair<FwdIter, FwdIter> >
                {
                    // propagate exceptions
                    f1.get(); f2.get();

                    hpx::future<FwdIter> f = r.call(p, non_seq(), first, last);
                    return f.then(
                        [=] (hpx::future<FwdIter> && f) mutable
                        ->  std::pair<FwdIter, FwdIter>
                        {
                            f.get();    // propagate exceptions
                            std::advance(first, std::distance(new_first, last));
                            return std::make_pair(first, last);
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
            {}

            template <typename ExPolicy, typename FwdIter>
            static IterPair
            sequential(ExPolicy, FwdIter first, FwdIter new_first,
                FwdIter last)
            {
                return sequential_rotate(first, new_first, last);
            }

            template <typename ExPolicy, typename FwdIter>
            static typename util::detail::algorithm_result<
                ExPolicy, IterPair
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter new_first,
                FwdIter last)
            {
                return util::detail::algorithm_result<
                        ExPolicy, IterPair
                    >::get(rotate_helper(std::forward<ExPolicy>(policy),
                            first, new_first, last));
            }
        };
        /// \endcond
    }

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
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a rotate algorithm invoked with
    /// an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \note The type of dereferenced \a FwdIter must meet the requirements
    ///       of \a MoveAssignable and \a MoveConstructible.
    ///
    /// \returns  The \a rotate algorithm returns a
    ///           \a hpx::future<tagged_pair<tag::begin(FwdIter), tag::end(FwdIter)> >
    ///           if the execution policy is of type
    ///           \a parallel_task_execution_policy and
    ///           returns \a tagged_pair<tag::begin(FwdIter), tag::end(FwdIter)>
    ///           otherwise.
    ///           The \a rotate algorithm returns the iterator equal to
    ///           pair(first + (last - new_first), last).
    ///
    template <typename ExPolicy, typename FwdIter,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<FwdIter>::value)>
    typename util::detail::algorithm_result<
        ExPolicy,
        hpx::util::tagged_pair<tag::begin(FwdIter), tag::end(FwdIter)>
    >::type
    rotate(ExPolicy && policy, FwdIter first, FwdIter new_first, FwdIter last)
    {
        static_assert(
            (hpx::traits::is_at_least_forward_iterator<FwdIter>::value),
            "Requires at least forward iterator.");

        typedef std::integral_constant<bool,
                is_sequential_execution_policy<ExPolicy>::value ||
                hpx::traits::is_forward_iterator<FwdIter>::value
            > is_seq;

        return hpx::util::make_tagged_pair<tag::begin, tag::end>(
            detail::rotate<std::pair<FwdIter, FwdIter> >().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, new_first, last));
    }

    ///////////////////////////////////////////////////////////////////////////
    // rotate_copy
    namespace detail
    {
        /// \cond NOINTERNAL

        // sequential rotate_copy
        template <typename FwdIter, typename OutIter>
        inline std::pair<FwdIter, OutIter>
        sequential_rotate_copy(FwdIter first, FwdIter new_first, FwdIter last,
            OutIter dest_first)
        {
            std::pair<FwdIter, OutIter> p1 =
                sequential_copy(new_first, last, dest_first);
            std::pair<FwdIter, OutIter> p2 =
                sequential_copy(first, new_first, std::move(p1.second));
            return std::make_pair(std::move(p1.first), std::move(p2.second));
        }

        template <typename ExPolicy, typename FwdIter, typename OutIter>
        hpx::future<std::pair<FwdIter, OutIter> >
        rotate_copy_helper(ExPolicy policy, FwdIter first,
            FwdIter new_first, FwdIter last, OutIter dest_first)
        {
            typedef std::false_type non_seq;

            parallel_task_execution_policy p =
                parallel_task_execution_policy()
                    .on(policy.executor())
                    .with(policy.parameters());

            typedef std::pair<FwdIter, OutIter> copy_return_type;

            hpx::future<copy_return_type> f =
                detail::copy<copy_return_type>().call(p, non_seq(),
                    new_first, last, dest_first);

            return f.then(
                [=](hpx::future<copy_return_type> && result)
                ->  hpx::future<copy_return_type>
                {
                    copy_return_type p1 = result.get();
                    return detail::copy<copy_return_type>().call(
                        p, non_seq(), first, new_first, p1.second);
                });
        }

        template <typename IterPair>
        struct rotate_copy
          : public detail::algorithm<rotate_copy<IterPair>, IterPair>
        {
            rotate_copy()
              : rotate_copy::algorithm("rotate_copy")
            {}

            template <typename ExPolicy, typename FwdIter, typename OutIter>
            static std::pair<FwdIter, OutIter>
            sequential(ExPolicy, FwdIter first, FwdIter new_first,
                FwdIter last, OutIter dest_first)
            {
                return sequential_rotate_copy(first, new_first, last, dest_first);
            }

            template <typename ExPolicy, typename FwdIter, typename OutIter>
            static typename util::detail::algorithm_result<
                ExPolicy, std::pair<FwdIter, OutIter>
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter new_first,
                FwdIter last, OutIter dest_first)
            {
                return util::detail::algorithm_result<ExPolicy, IterPair>::get(
                    rotate_copy_helper(std::forward<ExPolicy>(policy),
                        first, new_first, last, dest_first));
            }
        };
        /// \endcond
    }

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
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     bidirectional iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
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
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a rotate_copy algorithm invoked with
    /// an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a rotate_copy algorithm returns a
    ///           \a hpx::future<tagged_pair<tag::in(FwdIter), tag::out(OutIter)> >
    ///           if the execution policy is of type
    ///           \a parallel_task_execution_policy and
    ///           returns \a tagged_pair<tag::in(FwdIter), tag::out(OutIter)>
    ///           otherwise.
    ///           The \a rotate_copy algorithm returns the output iterator to the
    ///           element past the last element copied.
    ///
    template <typename ExPolicy, typename FwdIter, typename OutIter,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<FwdIter>::value &&
        hpx::traits::is_iterator<OutIter>::value)>
    typename util::detail::algorithm_result<
        ExPolicy,
        hpx::util::tagged_pair<tag::in(FwdIter), tag::out(OutIter)>
    >::type
    rotate_copy(ExPolicy && policy, FwdIter first, FwdIter new_first,
        FwdIter last, OutIter dest_first)
    {
        static_assert(
            (hpx::traits::is_at_least_forward_iterator<FwdIter>::value),
            "Requires at least forward iterator.");
        static_assert(
            (hpx::traits::is_output_iterator<OutIter>::value ||
                hpx::traits::is_at_least_input_iterator<OutIter>::value),
            "Requires at least output iterator.");

        typedef std::integral_constant<bool,
                is_sequential_execution_policy<ExPolicy>::value ||
                hpx::traits::is_output_iterator<OutIter>::value
            > is_seq;

        return hpx::util::make_tagged_pair<tag::in, tag::out>(
            detail::rotate_copy<std::pair<FwdIter, OutIter> >().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, new_first, last, dest_first));
    }
}}}

#endif
