//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/rotate.hpp

#if !defined(HPX_PARALLEL_DETAIL_ROTATE_AUG_05_2014_0138PM)
#define HPX_PARALLEL_DETAIL_ROTATE_AUG_05_2014_0138PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/unwrapped.hpp>
#include <hpx/lcos/local/dataflow.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/reverse.hpp>
#include <hpx/parallel/algorithms/copy.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>

#include <algorithm>
#include <iterator>

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

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
        FwdIter sequential_rotate(FwdIter first, FwdIter new_first, FwdIter last)
        {
            if (first != new_first && new_first != last)
                sequential_rotate_helper(first, new_first, last);

            std::advance(first, std::distance(new_first, last));
            return first;
        }

        template <typename ExPolicy, typename FwdIter>
        hpx::future<FwdIter>
        rotate_helper(ExPolicy policy, FwdIter first, FwdIter new_first,
            FwdIter last)
        {
            typedef boost::mpl::false_ non_seq;

            parallel_task_execution_policy p =
                parallel_task_execution_policy().with(policy.parameters());

            detail::reverse r;
            return lcos::local::dataflow(
                hpx::util::unwrapped([=]() mutable -> hpx::future<FwdIter>
                {
                    hpx::future<void> f = r.call(p, non_seq(), first, last);
                    std::advance(first, std::distance(new_first, last));
                    return f.then(
                        [first] (hpx::future<void> &&) -> FwdIter
                        {
                            return first;
                        });
                }),
                r.call(p, non_seq(), first, new_first),
                r.call(p, non_seq(), new_first, last));
        }

        template <typename FwdIter>
        struct rotate : public detail::algorithm<rotate<FwdIter>, FwdIter>
        {
            rotate()
              : rotate::algorithm("rotate")
            {}

            template <typename ExPolicy>
            static FwdIter
            sequential(ExPolicy, FwdIter first, FwdIter new_first,
                FwdIter last)
            {
                return sequential_rotate(first, new_first, last);
            }

            template <typename ExPolicy>
            static typename util::detail::algorithm_result<
                ExPolicy, FwdIter
            >::type
            parallel(ExPolicy policy, FwdIter first,
                FwdIter new_first, FwdIter last)
            {
                return util::detail::algorithm_result<ExPolicy, FwdIter>::get(
                    rotate_helper(policy, first, new_first, last));
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
    /// \returns  The \a rotate algorithm returns a \a hpx::future<FwdIter>
    ///           if the execution policy is of type
    ///           \a parallel_task_execution_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a rotate algorithm returns the iterator equal to
    ///           first + (last - new_first).
    ///
    template <typename ExPolicy, typename FwdIter>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    >::type
    rotate(ExPolicy && policy, FwdIter first, FwdIter new_first, FwdIter last)
    {
        typedef typename std::iterator_traits<FwdIter>::iterator_category
            iterator_category;

        static_assert(
            (boost::is_base_of<
                std::forward_iterator_tag, iterator_category>::value),
            "Required at least forward iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::forward_iterator_tag, iterator_category>
        >::type is_seq;

        return detail::rotate<FwdIter>().call(
            std::forward<ExPolicy>(policy), is_seq(), first, new_first, last);
    }

    ///////////////////////////////////////////////////////////////////////////
    // rotate_copy
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename ExPolicy, typename FwdIter, typename OutIter>
        hpx::future<OutIter>
        rotate_copy_helper(ExPolicy policy, FwdIter first,
            FwdIter new_first, FwdIter last, OutIter dest_first)
        {
            typedef boost::mpl::false_ non_seq;

            parallel_task_execution_policy p =
                parallel_task_execution_policy().with(policy.parameters());

            hpx::future<OutIter> f =
                detail::copy<OutIter>().call(p, non_seq(),
                    new_first, last, dest_first);

            return f.then(
                [=](hpx::future<OutIter> && it)
                {
                    return detail::copy<OutIter>().call(
                        p, non_seq(), first, new_first, it.get());
                });
        }

        template <typename OutIter>
        struct rotate_copy
          : public detail::algorithm<rotate_copy<OutIter>, OutIter>
        {
            rotate_copy()
              : rotate_copy::algorithm("rotate_copy")
            {}

            template <typename ExPolicy, typename FwdIter>
            static OutIter
            sequential(ExPolicy, FwdIter first, FwdIter new_first,
                FwdIter last, OutIter dest_first)
            {
                return std::rotate_copy(first, new_first, last, dest_first);
            }

            template <typename ExPolicy, typename FwdIter>
            static typename util::detail::algorithm_result<
                ExPolicy, OutIter
            >::type
            parallel(ExPolicy policy, FwdIter first, FwdIter new_first,
                FwdIter last, OutIter dest_first)
            {
                return util::detail::algorithm_result<ExPolicy, OutIter>::get(
                    rotate_copy_helper(policy, first, new_first, last, dest_first));
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
    /// \returns  The \a rotate_copy algorithm returns a \a hpx::future<OutIter>
    ///           if the execution policy is of type
    ///           \a parallel_task_execution_policy and
    ///           returns \a OutIter otherwise.
    ///           The \a rotate_copy algorithm returns the output iterator to the
    ///           element past the last element copied.
    ///
    template <typename ExPolicy, typename FwdIter, typename OutIter>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    rotate_copy(ExPolicy && policy, FwdIter first, FwdIter new_first,
        FwdIter last, OutIter dest_first)
    {
        typedef typename std::iterator_traits<FwdIter>::iterator_category
            forward_iterator_category;
        typedef typename std::iterator_traits<OutIter>::iterator_category
            output_iterator_category;

        static_assert(
            (boost::is_base_of<
                std::forward_iterator_tag, forward_iterator_category>::value),
            "Required at least forward iterator.");

        static_assert(
            (boost::mpl::or_<
                boost::is_base_of<
                    std::forward_iterator_tag, output_iterator_category>,
                boost::is_same<
                    std::output_iterator_tag, output_iterator_category>
            >::value),
            "Requires at least output iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::output_iterator_tag, output_iterator_category>
        >::type is_seq;

        return detail::rotate_copy<OutIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, new_first, last, dest_first);
    }
}}}

#endif
