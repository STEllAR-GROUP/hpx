//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/reverse.hpp

#if !defined(HPX_PARALLEL_DETAIL_REVERSE_JUL_29_2014_0432PM)
#define HPX_PARALLEL_DETAIL_REVERSE_JUL_29_2014_0432PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/unused.hpp>
#include <hpx/util/void_guard.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/is_negative.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/algorithms/copy.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <iterator>

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // reverse
    namespace detail
    {
        /// \cond NOINTERNAL
        struct reverse : public detail::algorithm<reverse>
        {
            reverse()
              : reverse::algorithm("reverse")
            {}

            template <typename ExPolicy, typename BidirIter>
            static hpx::util::unused_type
            sequential(ExPolicy, BidirIter first, BidirIter last)
            {
                std::reverse(first, last);
                return hpx::util::unused;
            }

            template <typename ExPolicy, typename BidirIter>
            static typename util::detail::algorithm_result<ExPolicy>::type
            parallel(ExPolicy policy, BidirIter first, BidirIter last)
            {
                typedef std::reverse_iterator<BidirIter> destination_iterator;
                typedef hpx::util::zip_iterator<BidirIter, destination_iterator>
                    zip_iterator;
                typedef typename zip_iterator::reference reference;
                typedef typename util::detail::algorithm_result<ExPolicy>::type
                    result_type;

                return hpx::util::void_guard<result_type>(),
                    for_each_n<zip_iterator>().call(
                        policy, boost::mpl::false_(),
                        hpx::util::make_zip_iterator(
                            first, destination_iterator(last)),
                        std::distance(first, last) / 2,
                        [](reference t) {
                            using hpx::util::get;
                            std::swap(get<0>(t), get<1>(t));
                        });
            }
        };
        /// \endcond
    }

    /// Reverses the order of the elements in the range [first, last).
    /// Behaves as if applying std::iter_swap to every pair of iterators
    /// first+i, (last-i) - 1 for each non-negative i < (last-first)/2.
    ///
    /// \note   Complexity: Linear in the distance between \a first and \a last.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam BidirIter  The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     bidirectional iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    ///
    /// The assignments in the parallel \a reverse algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a reverse algorithm invoked with
    /// an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a reverse algorithm returns a \a hpx::future<void>
    ///           if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a void otherwise.
    ///
    template <typename ExPolicy, typename BidirIter>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy>::type
    >::type
    reverse(ExPolicy && policy, BidirIter first, BidirIter last)
    {
        typedef typename std::iterator_traits<BidirIter>::iterator_category
            iterator_category;

        static_assert(
            (boost::is_base_of<
                std::bidirectional_iterator_tag, iterator_category>::value),
            "Required at least bidirectional iterator.");

        typedef typename is_sequential_execution_policy<ExPolicy>::type is_seq;

        return detail::reverse().call(
            std::forward<ExPolicy>(policy), is_seq(), first, last);
    }

    ///////////////////////////////////////////////////////////////////////////
    // reverse_copy
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename OutputIter>
        struct reverse_copy
          : public detail::algorithm<reverse_copy<OutputIter>, OutputIter>
        {
            reverse_copy()
              : reverse_copy::algorithm("reverse_copy")
            {}

            template <typename ExPolicy, typename BidirIter>
            static OutputIter
            sequential(ExPolicy, BidirIter first, BidirIter last,
                OutputIter dest_first)
            {
                return std::reverse_copy(first, last, dest_first);
            }

            template <typename ExPolicy, typename BidirIter>
            static typename util::detail::algorithm_result<
                ExPolicy, OutputIter
            >::type
            parallel(ExPolicy policy, BidirIter first, BidirIter last,
                OutputIter dest_first)
            {
                typedef std::reverse_iterator<BidirIter> iterator;

                return detail::copy<OutputIter>().call(
                    policy, boost::mpl::false_(),
                    iterator(last), iterator(first), dest_first);
            }
        };
        /// \endcond
    }

    /// Copies the elements from the range [first, last) to another range
    /// beginning at dest_first in such a way that the elements in the new
    /// range are in reverse order.
    /// Behaves as if by executing the assignment
    /// *(d_first + (last - first) - 1 - i) = *(first + i) once for each
    /// non-negative i < (last - first)
    /// If the source and destination ranges (that is, [first, last) and
    /// [dest_first, ddest_first+(last-first)) respectively) overlap, the
    /// behavior is undefined.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam BidirIter  The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     bidirectional iterator.
    /// \tparam OutputIter  The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest_first   Refers to the begin of the destination range.
    ///
    /// The assignments in the parallel \a reverse_copy algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a reverse_copy algorithm invoked with
    /// an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a reverse_copy algorithm returns a \a hpx::future<OutputIter>
    ///           if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a OutputIter otherwise.
    ///           The \a reverse_copy algorithm returns the output iterator to the
    ///           element past the last element copied.
    ///
    template <typename ExPolicy, typename BidirIter, typename OutputIter>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, OutputIter>::type
    >::type
    reverse_copy(ExPolicy && policy, BidirIter first, BidirIter last,
        OutputIter dest_first)
    {
        typedef typename std::iterator_traits<BidirIter>::iterator_category
            input_iterator_category;
        typedef typename std::iterator_traits<OutputIter>::iterator_category
            output_iterator_category;

        static_assert(
            (boost::is_base_of<
                std::bidirectional_iterator_tag, input_iterator_category>::value),
            "Required at least bidirectional iterator.");

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

        return detail::reverse_copy<OutputIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, dest_first);
    }
}}}

#endif
