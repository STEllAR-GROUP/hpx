//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/move.hpp

#if !defined(HPX_PARALLEL_DETAIL_MOVE_JUNE_16_2014_1106AM)
#define HPX_PARALLEL_DETAIL_MOVE_JUNE_16_2014_1106AM

#include <hpx/config.hpp>
#include <hpx/traits/is_iterator.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // move
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename OutIter>
        struct move : public detail::algorithm<move<OutIter>, OutIter>
        {
            move()
              : move::algorithm("move")
            {}

            template <typename ExPolicy, typename InIter>
            static OutIter
            sequential(ExPolicy, InIter first, InIter last, OutIter dest)
            {
                return std::move(first, last, dest);
            }

            template <typename ExPolicy, typename FwdIter>
            static typename util::detail::algorithm_result<
                ExPolicy, OutIter
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last,
                OutIter dest)
            {
                typedef hpx::util::zip_iterator<FwdIter, OutIter> zip_iterator;
                typedef typename zip_iterator::reference reference;
                typedef typename util::detail::algorithm_result<
                        ExPolicy, OutIter
                    >::type result_type;

                return get_iter<1, result_type>(
                    for_each_n<zip_iterator>().call(
                        std::forward<ExPolicy>(policy), std::false_type(),
                        hpx::util::make_zip_iterator(first, dest),
                        std::distance(first, last),
                        [](reference t) {
                            using hpx::util::get;
                            get<1>(t) = std::move(get<0>(t)); //-V573
                        }));
            }
        };
        /// \endcond
    }

    /// Moves the elements in the range [first, last), to another range
    /// beginning at \a dest. After this operation the elements in the
    /// moved-from range will still contain valid values of the appropriate
    /// type, but not necessarily the same values as before the move.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first move assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the move assignments.
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam OutIter     The type of the iterator representing the
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
    /// \param dest         Refers to the beginning of the destination range.
    ///
    /// The move assignments in the parallel \a move algorithm invoked
    /// with an execution policy object of type
    /// \a sequential_execution_policy execute in sequential order in
    /// the calling thread.
    ///
    /// The move assignments in the parallel \a move algorithm invoked
    /// with an execution policy object of type
    /// \a parallel_execution_policy or \a parallel_task_execution_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a move algorithm returns a \a hpx::future<OutIter> if
    ///           the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a OutIter otherwise.
    ///           The \a move algorithm returns the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename InIter, typename OutIter>
    inline typename std::enable_if<
        is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    move(ExPolicy && policy, InIter first, InIter last, OutIter dest)
    {
        static_assert(
            (hpx::traits::is_at_least_input_iterator<InIter>::value),
            "Requires at least input iterator.");
        static_assert(
            (hpx::traits::is_output_iterator<OutIter>::value ||
                hpx::traits::is_at_least_input_iterator<OutIter>::value),
            "Requires at least output iterator.");

        typedef std::integral_constant<bool,
                is_sequential_execution_policy<ExPolicy>::value ||
                hpx::traits::is_input_iterator<InIter>::value ||
                hpx::traits::is_output_iterator<OutIter>::value
            > is_seq;

        return detail::move<OutIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, dest);
    }
}}}

#endif
