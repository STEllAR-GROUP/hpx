//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/set_difference.hpp

#if !defined(HPX_PARALLEL_ALGORITHM_SET_DIFFERENCE_MAR_10_2015_0158PM)
#define HPX_PARALLEL_ALGORITHM_SET_DIFFERENCE_MAR_10_2015_0158PM

#include <hpx/config.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/decay.hpp>

#include <hpx/parallel/algorithms/copy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/set_operation.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/loop.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // set_difference
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename OutIter>
        struct set_difference
          : public detail::algorithm<set_difference<OutIter>, OutIter>
        {
            set_difference()
              : set_difference::algorithm("set_difference")
            {}

            template <typename ExPolicy, typename InIter1, typename InIter2,
                typename F>
            static OutIter
            sequential(ExPolicy, InIter1 first1, InIter1 last1,
                InIter2 first2, InIter2 last2, OutIter dest, F && f)
            {
                return std::set_difference(first1, last1, first2, last2, dest,
                    std::forward<F>(f));
            }

            template <typename ExPolicy, typename RanIter1, typename RanIter2,
                typename F>
            static typename util::detail::algorithm_result<
                ExPolicy, OutIter
            >::type
            parallel(ExPolicy && policy, RanIter1 first1, RanIter1 last1,
                RanIter2 first2, RanIter2 last2, OutIter dest, F && f)
            {
                typedef typename std::iterator_traits<RanIter1>::difference_type
                    difference_type1;
                typedef typename std::iterator_traits<RanIter2>::difference_type
                    difference_type2;

                if (first1 == last1)
                {
                    typedef util::detail::algorithm_result<
                            ExPolicy, OutIter
                        > result;
                    return result::get(std::move(dest));
                }

                if (first2 == last2)
                {
                    return util::detail::convert_to_result(
                        detail::copy<std::pair<RanIter1, OutIter> >()
                            .call(
                                std::forward<ExPolicy>(policy),
                                std::false_type(), first1, last1, dest
                            ),
                            [](std::pair<RanIter1, OutIter> const& p) -> OutIter
                            {
                                return p.second;
                            });
                }

                typedef typename set_operations_buffer<OutIter>::type buffer_type;
                typedef typename hpx::util::decay<F>::type func_type;

                return set_operation(std::forward<ExPolicy>(policy),
                    first1, last1, first2, last2, dest, std::forward<F>(f),
                    // calculate approximate destination index
                    [](difference_type1 idx1, difference_type2 idx2)
                    {
                        return idx1;
                    },
                    // perform required set operation for one chunk
                    [](RanIter1 part_first1, RanIter1 part_last1,
                        RanIter2 part_first2, RanIter2 part_last2,
                        buffer_type* dest, func_type const& f)
                    {
                        return std::set_difference(part_first1, part_last1,
                            part_first2, part_last2, dest, f);
                    });
            }
        };
        /// \endcond
    }

    /// Constructs a sorted range beginning at dest consisting of all elements
    /// present in the range [first1, last1) and not present in the range
    /// [first2, last2). This algorithm expects both input ranges to be sorted
    /// with the given binary predicate \a f.
    ///
    /// \note   Complexity: At most 2*(N1 + N2 - 1) comparisons, where \a N1 is
    ///         the length of the first sequence and \a N2 is the length of the
    ///         second sequence.
    ///
    /// Equivalent elements are treated individually, that is, if some element
    /// is found \a m times in [first1, last1) and \a n times in
    /// [first2, last2), it will be copied to \a dest exactly std::max(m-n, 0)
    /// times. The resulting range cannot overlap with either of the input
    /// ranges.
    ///
    /// The resulting range cannot overlap with either of the input ranges.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a set_difference requires \a Pred to meet
    ///                     the requirements of \a CopyConstructible. This defaults
    ///                     to std::less<>
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last1        Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the sequence of elements
    ///                     of the second range the algorithm will be applied to.
    /// \param last2        Refers to the end of the sequence of elements of
    ///                     the second range the algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param op           The binary predicate which returns true if the
    ///                     elements should be treated as equal. The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type1 must be such
    ///                     that objects of type \a InIter can
    ///                     be dereferenced and then implicitly converted to
    ///                     \a Type1
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with a sequential execution policy object execute in sequential
    /// order in the calling thread (\a sequenced_policy) or in a
    /// single new thread spawned from the current thread
    /// (for \a sequenced_task_policy).
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a set_difference algorithm returns a \a hpx::future<OutIter>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a OutIter otherwise.
    ///           The \a set_difference algorithm returns the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename Pred = detail::less>
    inline typename std::enable_if<
        execution::is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    set_difference(ExPolicy && policy, InIter1 first1, InIter1 last1,
        InIter2 first2, InIter2 last2, OutIter dest, Pred && op = Pred())
    {
        static_assert(
            (hpx::traits::is_input_iterator<InIter1>::value),
            "Requires at least input iterator.");
        static_assert(
            (hpx::traits::is_input_iterator<InIter2>::value),
            "Requires at least input iterator.");
        static_assert(
            (hpx::traits::is_output_iterator<OutIter>::value ||
                hpx::traits::is_input_iterator<OutIter>::value),
            "Requires at least output iterator.");

        typedef std::integral_constant<bool,
                execution::is_sequential_execution_policy<ExPolicy>::value ||
               !hpx::traits::is_random_access_iterator<InIter1>::value ||
               !hpx::traits::is_random_access_iterator<InIter2>::value ||
               !hpx::traits::is_random_access_iterator<OutIter>::value
            > is_seq;

        return detail::set_difference<OutIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first1, last1, first2, last2, dest, std::forward<Pred>(op));
    }
}}}

#endif
