//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/detail/rotate.hpp

#if !defined(HPX_PARALLEL_DETAIL_ROTATE_AUG_05_2014_0138PM)
#define HPX_PARALLEL_DETAIL_ROTATE_AUG_05_2014_0138PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/unwrapped.hpp>
#include <hpx/lcos/local/dataflow.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/detail/algorithm_result.hpp>
#include <hpx/parallel/detail/dispatch.hpp>
#include <hpx/parallel/detail/reverse.hpp>
#include <hpx/parallel/detail/copy.hpp>

#include <algorithm>
#include <iterator>

#include <boost/static_assert.hpp>
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
        hpx::future<FwdIter>
        rotate_helper(FwdIter first, FwdIter new_first, FwdIter last)
        {
            typedef boost::mpl::false_ non_seq;

            detail::reverse r;
            return lcos::local::dataflow(
                hpx::util::unwrapped([=]() mutable {
                    hpx::future<void> f = r.call(task, first, last, non_seq());
                    std::advance(first, std::distance(new_first, last));
                    return f.then(
                        [first](hpx::future<void> &&)
                        {
                            return first;
                        });
                }),
                r.call(task, first, new_first, non_seq()),
                r.call(task, new_first, last, non_seq()));
        }

        template <typename FwdIter>
        struct rotate : public detail::algorithm<rotate<FwdIter>, FwdIter>
        {
            rotate()
              : rotate::algorithm("rotate")
            {}

            template <typename ExPolicy>
            static FwdIter
            sequential(ExPolicy const&, FwdIter first, FwdIter new_first,
                FwdIter last)
            {
                return std::rotate(first, new_first, last);
            }

            template <typename ExPolicy>
            static typename detail::algorithm_result<ExPolicy, FwdIter>::type
            parallel(ExPolicy const& policy, FwdIter first,
                FwdIter new_first, FwdIter last)
            {
                return detail::algorithm_result<ExPolicy, FwdIter>::get(
                    rotate_helper(first, new_first, last));
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
    /// \a task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \note The type of dereferenced \a FwdIter must meet the requirements
    ///       of \a MoveAssignable and \a MoveConstructible.
    ///
    /// \returns  The \a rotate algorithm returns a \a hpx::future<FwdIter>
    ///           if the execution policy is of type \a task_execution_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a rotate algorithm returns the iterator equal to
    ///           first + (last - new_first).
    ///
    template <typename ExPolicy, typename FwdIter>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, FwdIter>::type
    >::type
    rotate(ExPolicy && policy, FwdIter first, FwdIter new_first, FwdIter last)
    {
        typedef typename std::iterator_traits<FwdIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::forward_iterator_tag, iterator_category>::value),
            "Required at least forward iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::forward_iterator_tag, iterator_category>
        >::type is_seq;

        return detail::rotate<FwdIter>().call(
            std::forward<ExPolicy>(policy), first, new_first, last, is_seq());
    }

    ///////////////////////////////////////////////////////////////////////////
    // rotate_copy
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename T>
        T wait_if_future(T && t)
        {
            return std::move(t);
        }

        template <typename T>
        T wait_if_future(hpx::future<T> && t)
        {
            return t.get();
        }

        template <typename OutputIter>
        struct rotate_copy
          : public detail::algorithm<rotate_copy<OutputIter>, OutputIter>
        {
            rotate_copy()
              : rotate_copy::algorithm("rotate_copy")
            {}

            template <typename ExPolicy, typename FwdIter>
            static OutputIter
            sequential(ExPolicy const&, FwdIter first, FwdIter new_first,
                FwdIter last, OutputIter dest_first)
            {
                return std::rotate_copy(first, new_first, last, dest_first);
            }

            template <typename ExPolicy, typename FwdIter>
            static typename detail::algorithm_result<ExPolicy, OutputIter>::type
            parallel(ExPolicy const& policy, FwdIter first, FwdIter new_first,
                FwdIter last, OutputIter dest_first)
            {
                typedef boost::mpl::false_ non_seq;
                copy<OutputIter> c;

                auto outiter = wait_if_future(
                    c.call(policy, new_first, last, dest_first, non_seq()));
                return detail::algorithm_result<ExPolicy, OutputIter>::get(
                    c.call(policy, first, new_first, outiter, non_seq()));
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
    /// \tparam OutputIter  The type of the iterator representing the
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
    /// \a task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a rotate_copy algorithm returns a \a hpx::future<OutputIter>
    ///           if the execution policy is of type \a task_execution_policy and
    ///           returns \a OutputIter otherwise.
    ///           The \a rotate_copy algorithm returns the output iterator to the
    ///           element past the last element copied.
    ///
    template <typename ExPolicy, typename FwdIter, typename OutputIter>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, OutputIter>::type
    >::type
    rotate_copy(ExPolicy && policy, FwdIter first, FwdIter new_first,
        FwdIter last, OutputIter dest_first)
    {
        typedef typename std::iterator_traits<FwdIter>::iterator_category
            forward_iterator_category;
        typedef typename std::iterator_traits<OutputIter>::iterator_category
            output_iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::forward_iterator_tag, forward_iterator_category>::value),
            "Required at least forward iterator.");

        BOOST_STATIC_ASSERT_MSG(
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

        return detail::rotate_copy<OutputIter>().call(
            std::forward<ExPolicy>(policy),
            first, new_first, last, dest_first, is_seq());
    }
}}}

#endif
