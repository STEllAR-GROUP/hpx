//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/detail/move_backward.hpp

#if !defined(HPX_PARALLEL_DETAIL_MOVE_BACKWARD_JUL_29_2014_0638PM)
#define HPX_PARALLEL_DETAIL_MOVE_BACKWARD_JUL_29_2014_0638PM

#include <hpx/hpx_fwd.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/detail/algorithm_result.hpp>
#include <hpx/parallel/detail/dispatch.hpp>
#include <hpx/parallel/detail/for_each.hpp>
#include <hpx/parallel/detail/move.hpp>
#include <hpx/parallel/detail/reverse_base.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <iterator>

#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // move_backward
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename BidirIter1, typename BidirIter2>
        BidirIter2 sequential_move_backward(BidirIter1 first, BidirIter1 last,
            BidirIter2 dest_last)
        {
            while (first != last)
                *--dest_last = std::move(*--last);
            return dest_last;
        }

        template <typename BidirIter2>
        struct move_backward
          : public detail::algorithm<move_backward<BidirIter2>, BidirIter2>
        {
            move_backward()
              : move_backward::algorithm("move_backward")
            {}

            template <typename ExPolicy, typename BidirIter1>
            static BidirIter2
            sequential(ExPolicy const&, BidirIter1 first, BidirIter1 last,
                BidirIter2 dest_last)
            {
                return sequential_move_backward(first, last, dest_last);
            }

            template <typename ExPolicy, typename BidirIter1>
            static typename detail::algorithm_result<ExPolicy, BidirIter2>::type
            parallel(ExPolicy const& policy, BidirIter1 first, BidirIter1 last,
                BidirIter2 dest_last)
            {
                typedef std::reverse_iterator<BidirIter1> source_iterator;
                typedef std::reverse_iterator<BidirIter2> destination_iterator;

                return reverse_base(
                    detail::move<destination_iterator>().call(
                        policy, source_iterator(last), source_iterator(first),
                        destination_iterator(dest_last),
                        boost::mpl::false_()));
            }
        };
        /// \endcond
    }

    /// Moves the elements from the range \a [first, last), to another range
    /// ending at \a dest_last. The elements are moved in reverse order (the last
    /// element is moved first), but their relative order is preserved.
    /// The behavior is undefined if \a dest_last is within \a (first, last].
    /// \a move must be used instead of \a move_backward in that case.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first move assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the move_backward assignments.
    /// \tparam BidirIter1  The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     bidirectional iterator.
    /// \tparam BidirIter2  The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     bidirectional iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param desr_last    Refers to the end of the destination range.
    ///
    /// The move assignments in the parallel \a move_backward algorithm invoked
    /// with an execution policy object of type
    /// \a sequential_execution_policy execute in sequential order in
    /// the calling thread.
    ///
    /// The move assignments in the parallel \a move_backward algorithm invoked
    /// with an execution policy object of type
    /// \a parallel_execution_policy or \a task_execution_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a move_backward algorithm returns a \a hpx::future<BidirIter2>
    ///           if the execution policy is of type \a task_execution_policy and
    ///           returns \a BidirIter2 otherwise.
    ///           The \a move_backward algorithm returns the iterator to the
    ///           last element copied.
    ///
    template <typename ExPolicy, typename BidirIter1, typename BidirIter2>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, BidirIter2>::type
    >::type
    move_backward(ExPolicy && policy, BidirIter1 first, BidirIter1 last,
        BidirIter2 dest_last)
    {
        typedef typename std::iterator_traits<BidirIter1>::iterator_category
            category1;
        typedef typename std::iterator_traits<BidirIter2>::iterator_category
            category2;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<std::input_iterator_tag, category1>::value),
            "Required at least bidirectional iterator.");
        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<std::input_iterator_tag, category2>::value),
            "Required at least bidirectional iterator.");

        typedef typename is_sequential_execution_policy<ExPolicy>::type is_seq;

        return detail::move_backward<BidirIter2>().call(
            std::forward<ExPolicy>(policy),
            first, last, dest_last, is_seq());
    }
}}}

#endif
