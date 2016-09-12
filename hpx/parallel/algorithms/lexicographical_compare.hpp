//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/lexicographical_compare.hpp

#if !defined(HPX_PARALLEL_DETAIL_LEXI_COMPARE_DEC_30_2014_0312PM)
#define HPX_PARALLEL_DETAIL_LEXI_COMPARE_DEC_30_2014_0312PM

#include <hpx/config.hpp>
#include <hpx/traits/is_iterator.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/predicates.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/algorithms/mismatch.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // lexicographical_compare
    namespace detail
    {
        /// \cond NOINTERNAL
        struct lexicographical_compare
            : public detail::algorithm<lexicographical_compare, bool>
        {
            lexicographical_compare()
              : lexicographical_compare::algorithm("lexicographical_compare")
            {}

           template <typename ExPolicy, typename InIter1, typename InIter2,
                typename Pred>
           static bool
           sequential(ExPolicy, InIter1 first1, InIter1 last1, InIter2 first2,
                InIter2 last2, Pred && pred)
            {
                return std::lexicographical_compare(first1, last1, first2, last2, pred);
            }

            template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
                typename Pred>
            static typename util::detail::algorithm_result<ExPolicy, bool>::type
            parallel(ExPolicy && policy, FwdIter1 first1, FwdIter1 last1,
                FwdIter2 first2, FwdIter2 last2, Pred && pred)
            {
                typedef hpx::util::zip_iterator<FwdIter1, FwdIter2> zip_iterator;
                typedef typename zip_iterator::reference reference;

                std::size_t count1 = std::distance(first1, last1);
                std::size_t count2 = std::distance(first2, last2);

                // An empty range is lexicographically less than any non-empty
                // range
                if (count1 == 0 && count2 != 0)
                {
                    return util::detail::algorithm_result<ExPolicy, bool>::get(
                        true);
                }

                if (count2 == 0 && count1 != 0)
                {
                    return util::detail::algorithm_result<ExPolicy, bool>::get(
                        false);
                }

                std::size_t count = (std::min)(count1, count2);
                util::cancellation_token<std::size_t> tok(count);

                using hpx::util::make_zip_iterator;
                return util::partitioner<ExPolicy, bool, void>::
                    call_with_index(
                        std::forward<ExPolicy>(policy),
                        make_zip_iterator(first1, first2), count, 1,
                        [pred, tok](zip_iterator it, std::size_t part_count,
                            std::size_t base_idx) mutable
                        {
                            util::loop_idx_n(
                                base_idx, it, part_count, tok,
                                [&pred, &tok](reference t, std::size_t i)
                                {
                                    using hpx::util::get;
                                    if (pred(get<0>(t), get<1>(t)) ||
                                        pred(get<1>(t), get<0>(t)))
                                    {
                                        tok.cancel(i);
                                    }
                                });
                        },
                        [=](std::vector<hpx::future<void> > &&) mutable -> bool
                        {
                            std::size_t mismatched = tok.get_data();

                            std::advance(first1, mismatched);
                            std::advance(first2, mismatched);

                            if (first1 != last1 && first2 != last2)
                                return pred(*first1, *first2);

                            return first2 != last2;
                        });
            }
        };
        /// \endcond
    }

    /// Checks if the first range [first1, last1) is lexicographically less than
    /// the second range [first2, last2). uses a provided predicate to compare
    /// elements.
    ///
    /// \note   Complexity: At most 2 * min(N1, N2) applications of the comparison
    ///         operation, where N1 = std::distance(first1, last)
    ///         and N2 = std::distance(first2, last2).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam InIter1     The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam InIter2     The type of the source iterators used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a lexicographical_compare requires \a Pred to
    ///                     meet the requirements of \a CopyConstructible. This defaults
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
    /// \param pred         Refers to the comparison function that the first
    ///                     and second ranges will be applied to
    ///
    /// The comparison operations in the parallel \a lexicographical_compare
    /// algorithm invoked with an execution policy object of type
    /// \a sequential_execution_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The comparison operations in the parallel \a lexicographical_compare
    /// algorithm invoked with an execution policy object of type
    /// \a parallel_execution_policy
    /// or \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \note     Lexicographical comparison is an operation with the
    ///           following properties
    ///             - Two ranges are compared element by element
    ///             - The first mismatching element defines which range
    ///               is lexicographically
    ///               \a less or \a greater than the other
    ///             - If one range is a prefix of another, the shorter range is
    ///               lexicographically \a less than the other
    ///             - If two ranges have equivalent elements and are of the same length,
    ///               then the ranges are lexicographically \a equal
    ///             - An empty range is lexicographically \a less than any non-empty
    ///               range
    ///             - Two empty ranges are lexicographically \a equal
    ///
    /// \returns  The \a lexicographically_compare algorithm returns a
    ///           \a hpx::future<bool> if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a bool otherwise.
    ///           The \a lexicographically_compare algorithm returns true
    ///           if the first range is lexicographically less, otherwise
    ///           it returns false.
    /// range [first2, last2), it returns false.
    ///
    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename Pred = detail::less>
    inline typename std::enable_if<
        is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, bool>::type
    >::type
    lexicographical_compare(ExPolicy && policy, InIter1 first1, InIter1 last1,
        InIter2 first2, InIter2 last2, Pred && pred = Pred())
    {
        static_assert(
            (hpx::traits::is_input_iterator<InIter1>::value),
            "Requires at least input iterator.");
        static_assert(
            (hpx::traits::is_input_iterator<InIter2>::value),
            "Requires at least input iterator.");

        typedef std::integral_constant<bool,
                is_sequential_execution_policy<ExPolicy>::value ||
               !hpx::traits::is_forward_iterator<InIter1>::value ||
               !hpx::traits::is_forward_iterator<InIter2>::value
            > is_seq;

        return detail::lexicographical_compare().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first1, last1, first2, last2, std::forward<Pred>(pred));
    }
}}}

#endif
